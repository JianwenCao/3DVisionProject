import torch

from .pinhole import proj_pinhole, iproj_pinhole
from lietorch import SE3, Sim3

MIN_DEPTH = 0.2

def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float(), indexing='ij')

    return torch.stack([x, y], dim=-1)

def actp(Gij, X0, jacobian=False):
    """ action on point cloud """
    X1 = Gij[:,:,None,None] * X0
    
    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,
                o,  d,  o, -Z,  o,  X, 
                o,  o,  d,  Y, -X,  o,
                o,  o,  o,  o,  o,  o,
            ], dim=-1).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,  X,
                o,  d,  o, -Z,  o,  X,  Y,
                o,  o,  d,  Y, -X,  o,  Z,
                o,  o,  o,  o,  o,  o,  o
            ], dim=-1).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None

def projective_transform(poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False):
    """ map points from ii->jj """

    # inverse project
    X0, Jz = iproj_pinhole(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)   # (pinhole)

    # transform
    Gij = poses[:,jj] * poses[:,ii].inv()

    # Gij.data[:,ii==jj] = torch.as_tensor([-1.08171900e-01, -1.02207120e-03, -1.43436505e-04, -7.35491558e-04, 2.53017345e-03,  2.35093021e-03,  9.99993765e-01], device="cuda")
    X1, Ja = actp(Gij, X0, jacobian=jacobian)
    
    # project
    intr_proj = intrinsics[:,jj]
    x1, Jp = proj_pinhole(X1, intr_proj, jacobian=jacobian, return_depth=return_depth)   # (pinhole)

    # exclude points too close to camera
    valid = ((X1[...,2] > MIN_DEPTH) & (X0[...,2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None,None,None].adjT(Jj)

        Jz = Gij[:,:,None,None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid
