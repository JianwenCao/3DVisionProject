import torch
import torch.nn.functional as F
import geom.projective_ops as pops

class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        try:
            U = torch.linalg.cholesky(H)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except Exception as e:
            print(e)
            ctx.failed = True
            xs = torch.zeros_like(b)
            U = torch.zeros_like(H)

        return xs, U

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

def block_solve(H, b, ep=0.1, lm=0.0001):
    """ solve normal equations """
    B, N, _, D, _ = H.shape
    I = torch.eye(D).to(H.device)
    H = H + (ep + lm*H) * I

    H = H.permute(0,1,3,2,4)
    H = H.reshape(B, N*D, N*D)
    b = b.reshape(B, N*D, 1)

    x, _ = CholeskySolver.apply(H,b)
    return x.reshape(B, N, D)


def schur_solve(H, E, C, v, w, ep=0.1, lm=0.0001, sless=False):
    """ solve using shur complement """
    
    B, P, M, D, HW = E.shape
    H = H.permute(0,1,3,2,4).reshape(B, P*D, P*D)
    E = E.permute(0,1,3,2,4).reshape(B, P*D, M*HW)
    Q = (1.0 / C).view(B, M*HW, 1)

    # damping
    I = torch.eye(P*D).to(H.device)
    H = H + (ep + lm*H) * I
    
    v = v.reshape(B, P*D, 1)
    w = w.reshape(B, M*HW, 1)

    Et = E.transpose(1,2)
    S = H - torch.matmul(E, Q*Et)
    v = v - torch.matmul(E, Q*w)

    dx, L = CholeskySolver.apply(S, v)
    if sless:
        return dx.reshape(B, P, D)

    dz = Q * (w - Et @ dx)    
    dx = dx.reshape(B, P, D)
    dz = dz.reshape(B, M, HW)

    F = torch.linalg.inv(L) @ (E * Q[...,0])
    dzcov = torch.sum(torch.square(F), dim=1) + Q[...,0]
    dzcov = dzcov.reshape(M, HW)

    return dx, dz, dzcov

def schur_solve_mono_prior(C, w, Hs, Es, vs, ep=0.1, lm=0.0001, dzcov=False):
    """ solve using shur complement """
    D = Hs.shape[-1]
    B, M, HW = C.shape
    Q = (1.0 / C).view(B, M*HW, 1)
    w = w.reshape(B, M*HW, 1)

    H = Hs.permute(0,1,3,2,4).reshape(B, M*D, M*D)
    E = Es.permute(0,1,3,2,4).reshape(B, M*D, M*HW)
    v = vs.reshape(B, M*D, 1)

    # damping
    I = torch.eye(M*D).to(H.device)
    H = H + (ep + lm*H) * I
    
    Et = E.transpose(1,2)
    S = H - torch.matmul(E, Q*Et)
    v = v - torch.matmul(E, Q*w)

    dso, L = CholeskySolver.apply(S, v)
    dz = Q * (w - Et @ dso)
    dz = dz.reshape(B, M, HW)
    dso = dso.reshape(B, M, D)

    F = torch.linalg.inv(L) @ (E * Q[...,0])
    dzcov = torch.sum(torch.square(F), dim=1) + Q[...,0]
    # dzcov = Q[...,0]
    dzcov = dzcov.reshape(M, HW)
    return dso, dz, dzcov