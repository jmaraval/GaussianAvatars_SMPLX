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

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from tqdm import tqdm
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import pickle
from glob import glob
from math import atan2

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: Optional[np.array]
    image_path: str
    image_name: str
    width: int
    height: int
    bg: np.array = np.array([0, 0, 0])
    timestep: Optional[int] = None
    camera_id: Optional[int] = None
    fg_mask_path: Optional[str] = None

class CameraInfoProj(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    fy: np.array
    fx: np.array
    FovX: np.array
    FovY: np.array
    cx: np.array
    cy: np.array
    image: Optional[np.array]
    image_path: str
    image_name: str
    width: int
    height: int
    bg: np.array = np.array([0, 0, 0])
    timestep: Optional[int] = None
    camera_id: Optional[int] = None

class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    point_cloud: Optional[BasicPointCloud]
    ply_path: Optional[str]
    val_cameras: list = []
    train_meshes: dict = {}
    test_meshes: dict = {}
    tgt_train_meshes: dict = {}
    tgt_test_meshes: dict = {}

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        width, height = image.size

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if 'camera_angle_x' in contents:
            fovx_shared = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            file_path = frame["file_path"]
            # Append default extension only when the path has no extension at all.
            # (Avoids double-extension like .jpg.png when absolute paths are provided.)
            if not os.path.splitext(file_path)[1]:
                file_path += extension
            cam_name = os.path.join(path, file_path)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            
            if 'w' in frame and 'h' in frame:
                image = None
                width = frame['w']
                height = frame['h']
            else:
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                width, height = image.size

            if 'camera_angle_x' in frame:
                fovx = frame["camera_angle_x"]
            else:
                fovx = fovx_shared
            fovy = focal2fov(fov2focal(fovx, width), height)

            timestep = frame["timestep_index"] if 'timestep_index' in frame else None
            camera_id = frame["camera_index"] if 'camera_id' in frame else None

            # Foreground mask: resolve to absolute path if present
            fg_mask_path = frame.get("fg_mask_path", None)
            if fg_mask_path and not os.path.isabs(fg_mask_path):
                fg_mask_path = os.path.join(path, fg_mask_path)
            if fg_mask_path and not os.path.exists(fg_mask_path):
                fg_mask_path = None  # silently ignore missing masks

            cam_infos.append(CameraInfo(
                uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, bg=bg, image=image,
                image_path=image_path, image_name=image_name,
                width=width, height=height,
                timestep=timestep, camera_id=camera_id,
                fg_mask_path=fg_mask_path))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readMeshesFromTransforms(path, transformsfile):
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        
        mesh_infos = {}
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            if not 'timestep_index' in frame or frame["timestep_index"] in mesh_infos:
                continue

            flame_param = dict(np.load(os.path.join(path, frame['flame_param_path']), allow_pickle=True))
            mesh_infos[frame["timestep_index"]] = flame_param
    return mesh_infos

def readDynamicNerfInfo(path, white_background, eval, extension=".png", target_path=""):
    print("Reading Training Transforms")
    if target_path != "":
        train_cam_infos = readCamerasFromTransforms(target_path, "transforms_train.json", white_background, extension)
    else:
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    
    print("Reading Training Meshes")
    train_mesh_infos = readMeshesFromTransforms(path, "transforms_train.json")
    if target_path != "":
        print("Reading Target Meshes (Training Division)")
        tgt_train_mesh_infos = readMeshesFromTransforms(target_path, "transforms_train.json")
    else:
        tgt_train_mesh_infos = {}
    
    print("Reading Validation Transforms")
    if target_path != "":
        val_cam_infos = readCamerasFromTransforms(target_path, "transforms_val.json", white_background, extension)
    else:
        val_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension)
    
    print("Reading Test Transforms")
    if target_path != "":
        test_cam_infos = readCamerasFromTransforms(target_path, "transforms_test.json", white_background, extension)
    else:
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    print("Reading Test Meshes")
    test_mesh_infos = readMeshesFromTransforms(path, "transforms_test.json")
    if target_path != "":
        print("Reading Target Meshes (Test Division)")
        tgt_test_mesh_infos = readMeshesFromTransforms(target_path, "transforms_test.json")
    else:
        tgt_test_mesh_infos = {}
    
    if target_path != "" or not eval:
        train_cam_infos.extend(val_cam_infos)
        val_cam_infos = []
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        train_mesh_infos.update(test_mesh_infos)
        test_mesh_infos = {}

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           val_cameras=val_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None,
                           train_meshes=train_mesh_infos,
                           test_meshes=test_mesh_infos,
                           tgt_train_meshes=tgt_train_mesh_infos,
                           tgt_test_meshes=tgt_test_mesh_infos)
    return scene_info


def readCamerasFrompkl(pklfile, global_t, white_background, extension=".png"):
    cam_infos = []
    meshes = {}

    print("  loading:", pklfile)
    with open(pklfile, "rb") as f:
        try:
            results = pickle.load(f)
        except Exception as e:
            # Some pkls might be saved with protocol issues; re-raise with context
            raise RuntimeError(f"Failed to load {pklfile}: {e}")

        # detect which keys are present
        # try common names used in SignAvatar scripts
        focals = results['focal']
        princpts = results['princpt']
        smplx_all = results['smplx']

        # height/width if present
        height = results['height']
        width  = results['width']

        if smplx_all is None:
            raise AssertionError(f"No SMPL-X data found inside {pklfile}. Keys: {list(results.keys())}")

        smplx_all = np.array(smplx_all)  # (N, D) where D typically 182 in SignAvatar
        N = smplx_all.shape[0]

        # focal/princpt may be per-video or per-frame; normalize to (N,2) and (N,2)
        if focals is None:
            # fallback to virtual focal (use width/height heuristics)
            fx = fy = max(width, height) * 1.5 if (width and height) else 5000.0
            focals = np.tile(np.array([fx, fy], dtype=np.float32)[None, :], (N, 1))
        else:
            focals = np.array(focals)
            if focals.ndim == 1:
                focals = np.tile(focals[None, :], (N, 1))
            elif focals.shape[0] != N:
                # broadcast if single set provided
                focals = np.tile(focals[0:1, :], (N, 1))

        if princpts is None:
            px = width / 2.0 if width else 0.0
            py = height / 2.0 if height else 0.0
            princpts = np.tile(np.array([px, py], dtype=np.float32)[None, :], (N, 1))
        else:
            princpts = np.array(princpts)
            if princpts.ndim == 1:
                princpts = np.tile(princpts[None, :], (N, 1))
            elif princpts.shape[0] != N:
                princpts = np.tile(princpts[0:1, :], (N, 1))

        # If dataset provides a list of valid frame indices, use it, otherwise assume 0..N-1
        valid_indices = results.get('total_valid_index', None)
        if valid_indices is None:
            indices = list(range(1,N))
        else:
            indices = list(valid_indices)

        # prepare images folder (if any)
        pattern = r"^(?P<IDVID>[^_]+)_(?P<num_sentence>\d+)-(?P<IDSIGNER>[^-]+)-rgb_front\.pkl$"
        import re
        match = re.match(pattern, os.path.basename(pklfile))
        if not match:
            raise ValueError(f"Filename does not match expected format: {os.path.basename(pklfile)}")
        id_vid =match.group("IDVID")
        num_sentence = match.group("num_sentence")
        id_signer = match.group("IDSIGNER")

        images_folder = os.path.join("/home/jmaraval/Documents/GaussianAvatars/data/SignAvatars/how2sign/", id_signer,id_vid, num_sentence, "images")
        images_exist = os.path.isdir(images_folder)

        # iterate frames in this pkl
        for local_idx, frame_idx in enumerate(indices):
            # handle case where results['smplx'] is N x D and indices refers to indices into it
            smplx_vec = smplx_all[local_idx] if local_idx < smplx_all.shape[0] else smplx_all[frame_idx]

            # unpack vector according to SignAvatar snippet:
            # 0:3 global orient
            # 3:66 body_pose (63)
            # 66:111 left hand (45)
            # 111:156 right hand (45)
            # 156:159 jaw (3)
            # 159:169 betas (10)
            # 169:179 expression (10)
            # 179:182 cam_trans (3)
            if smplx_vec.size < 182:
                # fallback: maybe the vector is already a dict per-frame
                if isinstance(smplx_vec, dict):
                    # try to use explicit keys
                    global_orient = np.array(smplx_vec.get('global_orient', np.zeros(3)), dtype=np.float32)
                    body_pose = np.array(smplx_vec.get('body_pose', np.zeros(63)), dtype=np.float32)
                    left_hand = np.array(smplx_vec.get('left_hand_pose', np.zeros(45)), dtype=np.float32)
                    right_hand = np.array(smplx_vec.get('right_hand_pose', np.zeros(45)), dtype=np.float32)
                    jaw = np.array(smplx_vec.get('jaw_pose', np.zeros(3)), dtype=np.float32)
                    betas = np.array(smplx_vec.get('betas', np.zeros(10)), dtype=np.float32)
                    expr = np.array(smplx_vec.get('expression', np.zeros(10)), dtype=np.float32)
                    cam_trans = np.array(smplx_vec.get('transl', np.zeros(3)), dtype=np.float32)
                else:
                    raise AssertionError("Unexpected SMPL-X vector size: {}".format(smplx_vec.size))
            else:
                global_orient = np.array(smplx_vec[0:3], dtype=np.float32)
                body_pose = np.array(smplx_vec[3:66], dtype=np.float32)
                left_hand = np.array(smplx_vec[66:111], dtype=np.float32)
                right_hand = np.array(smplx_vec[111:156], dtype=np.float32)
                jaw = np.array(smplx_vec[156:159], dtype=np.float32)
                betas = np.array(smplx_vec[159:169], dtype=np.float32)
                expr = np.array(smplx_vec[169:179], dtype=np.float32)
                cam_trans = np.array(smplx_vec[179:182], dtype=np.float32)


            # intrinsics for this frame
            focal_frame = focals[local_idx]
            princpt_frame = princpts[local_idx]
            fx, fy = float(focal_frame[0]), float(focal_frame[1])
            cx, cy = float(princpt_frame[0]), float(princpt_frame[1])


            # image path if available: try by frame index naming convention (frame_idx.png)
            image_path = None
            image_obj = None
            if images_exist:
                # try a few filename patterns
                candidate_names = [
                    f"{frame_idx:05d}.png",
                    f"{frame_idx:06d}.png",
                    f"{local_idx:05d}.png",
                    f"{local_idx:06d}.png",
                    f"{frame_idx}.png",
                ]
                for name in candidate_names:
                    p = os.path.join(images_folder, name)
                    if os.path.exists(p):
                        image_path = p

            # Build CameraInfo — we set R as identity and T as cam_trans (camera-centered translation)

            R = np.eye(3, dtype=np.float32)
            #R = np.array([
            #    [1,  0,  0],
            #    [0, -1,  0],
            #    [0,  0, -1],
            #], dtype=np.float32)
            #T = cam_trans.astype(np.float32)
            T = np.zeros(3, dtype=np.float32)

            FoVx = focal2fov(fx, width)
            FoVy = focal2fov(fy, height)

            cam_uid = global_t
            cam_info = CameraInfoProj(
                uid=cam_uid,
                R=R,
                T=T,
                fy=fy,
                fx=fx,
                FovX=FoVx,
                FovY=FoVy,
                cx=cx,
                cy=cy,
                image=image_obj,
                image_path=image_path,
                image_name=os.path.basename(image_path) if image_path else f"{cam_uid}",
                width=width if width else 0,
                height=height if height else 0,
                bg=np.array([1,1,1]) if white_background else np.array([0,0,0]),
                timestep=global_t,
                camera_id=0
            )

            # assign to train/test depending on eval flag; here we keep simple: all as train unless eval True
            if eval and (len(cam_infos) < max(1, int(0.1 * (len(pklfile) * N)))):
                cam_infos.append(cam_info)
            else:
                cam_infos.append(cam_info)

            # Mesh param dict expected by SMPLXGaussianModel (numpy arrays)
            mesh_param = {
                "betas": betas.astype(np.float32),
                "expr": expr.astype(np.float32),
                "body_pose": body_pose.astype(np.float32),
                "jaw_pose": jaw.astype(np.float32),
                "left_hand_pose": left_hand.astype(np.float32),
                "right_hand_pose": right_hand.astype(np.float32),
                "rotation": global_orient.astype(np.float32),  # named 'rotation' to mimic flame naming in some code paths
                "global_orient": global_orient.astype(np.float32),
                "translation": cam_trans.astype(np.float32),
                # dynamic_offset not present — will be initialized by model
                "dynamic_offset": cam_trans.astype(np.float32),
            }

            # store in train_meshes by timestep
            meshes[global_t] = mesh_param

            global_t += 1

    return cam_infos, meshes, global_t

def readSignAvatarsSceneInfo(path, white_background, eval, extension=".png"):
    """
    Loader for SignAvatar-style SMPL-X `.pkl` files.
    Supports either:
      - path/  (containing many .pkl files, each per sequence)
      - a single .pkl file passed as `path`

    Produces a SceneInfo with train/test camera lists and per-frame smplx parameter dicts:
      train_meshes[timestep] = {'betas':..., 'expression':..., 'body_pose':..., 'jaw_pose':..., 'left_hand_pose':..., 'right_hand_pose':..., 'global_orient':..., 'translation':...}
    """
    print("Reading SignAvatar dataset at:", path)

    pkl_paths = []
    pkl_paths_test = None
    if os.path.isfile(path) and path.lower().endswith(".pkl"):
        pkl_paths = [path]
    elif os.path.isfile(os.path.join(path, "sequences_train.txt")):
        annot_path="/home/jmaraval/Documents/SignAvatars/datasets/language2motion/annotations/SMPL-X"
        with open(os.path.join(path, "sequences_train.txt")) as file:
            pkl_paths = [os.path.join(annot_path,line.rstrip()) for line in file]
        if os.path.isfile(os.path.join(path, "sequences_test.txt")):
            with open(os.path.join(path, "sequences_test.txt")) as file:
                pkl_paths_test = [os.path.join(annot_path,line.rstrip()) for line in file]
    else:
        raise AssertionError(f"No .pkl found at {path}")

    

    train_cam_infos = []
    val_cam_infos = []
    test_cam_infos = []
    train_meshes = {}
    test_meshes = {}

    # timestep counter across all pkls (keeps unique timesteps)
    global_t = 0

    for pkl_file in pkl_paths:
        cam_infos, meshes, global_t = readCamerasFrompkl(pkl_file, global_t, white_background, extension)
        train_cam_infos.extend(cam_infos)
        train_meshes.update(meshes)

    for pkl_file in pkl_paths_test:
        cam_infos, meshes, global_t = readCamerasFrompkl(pkl_file, global_t, white_background, extension)
        test_cam_infos.extend(cam_infos)
        test_meshes.update(meshes)
    
    # choose splits: if eval requested, use test_cam_infos else empty
#    if eval:
#        # if test list empty, move last 10% frames to test
#        if len(test_cam_infos) == 0 and len(train_cam_infos) > 0:
#            n_test = max(1, int(0.1 * len(train_cam_infos)))
#            test_cam_infos = train_cam_infos[-n_test:]
#            train_cam_infos = train_cam_infos[:-n_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path=None
    pcd=None
#    if not os.path.exists(ply_path):
#        # Since this data set has no colmap data, we start with random points
#        num_pts = 100_000
#        print(f"Generating random point cloud ({num_pts})...")
#        
#        # We create random points inside the bounds of the synthetic Blender scenes
#        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
#        shs = np.random.random((num_pts, 3)) / 255.0
#        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
#
#        storePly(ply_path, xyz, SH2RGB(shs) * 255)
#    try:
#        pcd = fetchPly(ply_path)
#    except:
#        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        val_cameras=val_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        train_meshes=train_meshes,
        test_meshes=test_meshes,
        tgt_train_meshes={},
        tgt_test_meshes={},
    )


    print(f"SignAvatar loader: {len(train_cam_infos)} train cams, {len(test_cam_infos)} test cams, {len(train_meshes)} mesh frames")
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "DynamicNerf" : readDynamicNerfInfo,
    "Blender" : readNerfSyntheticInfo,
    "SignAvatars": readSignAvatarsSceneInfo,
}