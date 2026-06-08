#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrix2, focal2fov, fov2focal
import copy

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, bg, image_width, image, image_height, image_path,
                 image_name, uid, cx=None, cy=None, trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 timestep=None, fg_mask_path=None, data_device="cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.bg = bg
        self.image = image
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.image_name = image_name
        self.timestep = timestep
        self.fg_mask_path = fg_mask_path
        self.cx = cx
        self.cy = cy

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale


        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)  #.cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)  #.cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def clone(self):
        return copy.deepcopy(self)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, timestep):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.timestep = timestep

class CameraProj(nn.Module):
    def __init__(self, colmap_id, R, T, fx, fy, bg, image_width, image, image_height, image_path,
                 image_name, uid, cx=None, cy=None, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 timestep=None, data_device = "cuda"
                 ):
        super(CameraProj, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = focal2fov(fx, image_height)
        self.FoVy = focal2fov(fy, image_width)
        self.bg = bg
        self.image = image
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.image_name = image_name
        self.timestep = timestep
        self.cx = cx
        self.cy = cy

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # Build projection matrix P such that NDC x = (2*fx/W) * X_cam/Z + (2*cx/W - 1)
        P = torch.zeros(4, 4, dtype=torch.float32)

        # scale terms
        P[0, 0] = 2.0 * fx / image_width
        P[1, 1] = 2.0 * fy / image_height

        # principal point offsets mapped to [-1,1]
        P[0, 2] = 2.0 * (cx / image_width) - 1.0
        # NOTE: many systems have image origin at top-left. If your cy is top-left origin,
        # you may need to use 1.0 - 2.0*(cy/H) or flip Y later. Keep this in mind.
        P[1, 2] = 2.0 * (cy / image_height) - 1.0

        # z mapping (OpenGL-ish)
        P[2, 2] = -(self.zfar + self.znear) / (self.zfar - self.znear)
        P[2, 3] = -2.0 * self.zfar * self.znear / (self.zfar - self.znear)
        P[3, 2] = -1.0

        # Convert P to the same orientation as your Camera expects (transpose in Camera ctor)
        self.projection_matrix = P.transpose(0, 1).clone()

        # Build world_view_transform matrix: we want a 4x4 where transform acts as:
        #  X_cam = R @ X_world + T
        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = np.array(R, dtype=np.float32)
        Rt[:3, 3] = np.array(T, dtype=np.float32)

        # Camera constructor previously used getWorld2View2(...) -> returned Rt transpose etc.
        # To be consistent, set cam.world_view_transform as the transpose (like original code).
        self.world_view_transform = torch.tensor(Rt, dtype=torch.float32).transpose(0, 1)

        # Full proj transform used by your renderer:
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        # other bookkeeping:
        self.camera_center = torch.tensor(np.linalg.inv(Rt)[:3, 3], dtype=torch.float32)

    def clone(self):
        return copy.deepcopy(self)
    

class CameraProj2(nn.Module):
    def __init__(self, colmap_id, R, T, fx, fy, bg, image_width, image, image_height, image_path,
                 image_name, uid, cx=None, cy=None, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 timestep=None, data_device = "cuda"
                 ):
        super(CameraProj2, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = focal2fov(fx, image_width)
        self.FoVy = focal2fov(fy, image_height)
        self.bg = bg
        self.image = image
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.image_name = image_name
        self.timestep = timestep
        self.cx = cx
        self.cy = cy

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale


        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)  #.cuda()
        self.projection_matrix = getProjectionMatrix2(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=self.cx, cy=self.cy, width=self.image_width, height=self.image_height).transpose(0,1)  #.cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def clone(self):
        return copy.deepcopy(self)