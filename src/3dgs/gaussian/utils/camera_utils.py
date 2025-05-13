import torch
from torch import nn
from gaussian.utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        normal,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.normal = normal
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_tracking(color, depth, normal, pose, idx, projection_matrix, K, tstamp=None):
        cam = Camera(
            idx,
            color,
            depth,
            normal,
            pose,
            projection_matrix,
            K[0],
            K[1],
            K[2],
            K[3],
            focal2fov(K[0], K[-2]),
            focal2fov(K[1], K[-1]),
            K[-1],
            K[-2])
        cam.R = pose[:3, :3]
        cam.T = pose[:3, 3]
        cam.tstamp = tstamp
        return cam
    
    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            None,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            # device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )
    
    @property
    def frustum_planes(self) -> torch.Tensor:
        """
        (6, 4) world-space frustum planes [a,b,c,d] in GS coords
        (x-right, y-down, z-forward).  Inside: a·x + b·y + c·z + d ≤ 0.
        """
        # Build (4,4) clip matrix
        clip = (self.projection_matrix @ self.world_view_transform).to(self.device)

        # transpose so we can treat the GL columns as our slicing axis
        m = clip.T  # now m[i] is the i’th CLIP *column*

        # extract the 6 planes: (col3 ± col0), (col3 ± col1), (col3 ± col2)
        planes = torch.empty((6,4), dtype=m.dtype, device=self.device)
        planes[0] = m[3] + m[0]   # left
        planes[1] = m[3] - m[0]   # right
        planes[2] = m[3] + m[1]   # bottom
        planes[3] = m[3] - m[1]   # top
        planes[4] = m[3] + m[2]   # near
        planes[5] = m[3] - m[2]   # far

        # normalize normals to unit length
        planes /= planes[:, :3].norm(dim=1, keepdim=True)

        # convert to GS axes (x→, y↓, z→)
        planes[:, :3] *= torch.tensor([1.0, -1.0, -1.0], device=self.device)

        # ensure near plane normal points toward camera (nz < 0)
        # and far plane normal points away  (nz > 0)
        if planes[4, 2] > 0: planes[4] *= -1
        if planes[5, 2] < 0: planes[5] *= -1

        return planes

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None
