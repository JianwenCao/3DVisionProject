import lietorch
import torch
import torch.nn.functional as F
import droid_backends

from .chol import block_solve, schur_solve, schur_solve_mono_prior
import geom.projective_ops as pops

from torch_scatter import scatter_sum


# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))


def BA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1):
    """ Full Bundle Adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Jz = Jz.reshape(B, N, ht*wd, -1)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
    Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

    w = w.view(B, N, ht*wd, -1)
    r = r.view(B, N, ht*wd, -1)
    wk = torch.sum(w*r*Jz, dim=-1)
    Ck = torch.sum(w*Jz*Jz, dim=-1)

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # only optimize keyframe poses
    P = P - fixedp
    ii = ii - fixedp
    jj = jj - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)

    C = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)

    C = C + eta.view(*C.shape) + 1e-7

    H = H.view(B, P, P, D, D)
    E = E.view(B, P, M, D, ht*wd)

    ### 3: solve the system ###
    dx, dz, dzcov = schur_solve(H, E, C, v, w)
    
    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)

    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.001)

    return poses, disps, dzcov


def MoBA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """ Motion only bundle adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    # only optimize keyframe poses
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)
    
    H = H.view(B, P, P, D, D)

    ### 3: solve the system ###
    dx = block_solve(H, v)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    return poses


def get_prior_depth_aligned(depth_prior, scales):
    M, ht, wd = depth_prior.shape
    hs, ws = scales.shape[-2:]
    meshx, meshy = torch.meshgrid(torch.linspace(0, hs-1-1e-6, ht), torch.linspace(0, ws-1-1e-6, wd), indexing='ij')
    grid = torch.stack((meshy, meshx), -1).cuda()
    grid = grid.unsqueeze(0).expand(M, -1, -1, -1).contiguous()
    mscales_bi, Jbi = droid_backends.bi_inter(scales, grid)
    depth_prior_aligned = depth_prior * mscales_bi
    return depth_prior_aligned, Jbi


def JDSA(target, weight, eta, poses, disps, intrinsics, disps_prior, dscales, ii, jj, alpha):

    B, P, ht, wd = disps.shape
    N = ii.shape[0]

    ### 1: commpute jacobians and residuals ###
    C, w = droid_backends.proj_trans(poses.data.squeeze(), disps[0], intrinsics[0], target, weight, ii, jj)

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    disps_prior = disps_prior[kx]
    m = (disps_prior > 0).to(torch.float).view(-1, ht*wd)

    hs, ws = dscales.shape[-2:]
    disps_bi, Jbi = get_prior_depth_aligned(disps_prior, dscales[kx])

    rd = (disps[0,kx] - disps_bi).view(-1, ht*wd)
    Jd = torch.ones_like(rd).view(1, -1, 1, ht*wd)
    # Jd = (-1. / (disps[0,kx] ** 2)).view(1, -1, 1, ht*wd)
    Jso = -m.unsqueeze(-1) * disps_prior.view(-1, ht*wd).unsqueeze(-1) * Jbi.view(M, ht*wd, -1)[None]

    alpha = torch.ones(M,ht*wd,1).float().cuda() * alpha

    D = hs*ws
    fixedp = kx[0]
    kx = kx - fixedp
    wJsoT = (alpha * Jso).transpose(2,3)
    Hs = safe_scatter_add_mat(wJsoT @ Jso, kx, kx, M, M).view(B, M, M, D, D)
    Es = safe_scatter_add_mat(wJsoT * Jd, kx, kx, M, M).view(B, M, M, D, ht*wd)
    vs = safe_scatter_add_vec(-wJsoT @ rd[None].unsqueeze(-1), kx, M)
    kx += fixedp

    alpha = alpha.squeeze()
    C = C[None] + m * alpha * (Jd * Jd).squeeze() + (1-m) * eta.view(*C.shape)
    w = w[None] - m * alpha * rd * Jd.squeeze()

    ### 3: solve the system ###
    dso, dz, dzcov = schur_solve_mono_prior(C, w, Hs, Es, vs, dzcov=True)

    ### 4: apply retraction ###
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
    dscales[kx] += dso.view(-1, hs, ws)

    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.001)

    return disps, dscales, dzcov
