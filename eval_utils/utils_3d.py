import cv2
import loguru
import matplotlib.pyplot as plt
import numpy as np 
import torch
import torch.nn.functional as F
import trimesh
from multipledispatch import dispatch
from packaging import version
from scipy.spatial.transform import Rotation
 
def to_array(x, dtype=float):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype)
    elif isinstance(x, list):
        return [to_array(a) for a in x]
    # elif isinstance(x, SafeDict):
    #     return SafeDict(to_array(dict(x)))
    elif isinstance(x, dict):
        return {k: to_array(v) for k, v in x.items()} 
    else:
        return x

def to_tensor(data, dtype=None, device=None):
    if isinstance(data, torch.Tensor):
        return data.clone().to(dtype=dtype, device=device)
    else:
        return torch.tensor(data, dtype=dtype, device=device)

def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    if isinstance(pts, np.ndarray):
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    else:
        ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=pts.device)
        pts_hom = torch.cat((pts, ones), dim=1)
    return pts_hom


def hom_to_cart(pts):
    """
    :param pts: (N, 4 or 3)
    :return pts_hom: (N, 3 or 2)
    """
    return pts[:, :-1] / pts[:, -1:]


@dispatch(np.ndarray, np.ndarray)
def canonical_to_camera(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :param calib:Calibration
    :return:
    """
    pts = cart_to_hom(pts)
    # pts = np.hstack((pts, np.ones((pts.shape[0], 1))))  # 4XN
    pts = pts @ pose.T  # 4xN
    pts = hom_to_cart(pts)
    return pts


@dispatch(np.ndarray, np.ndarray)
def canonical_to_camera(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :param calib:Calibration
    :return:
    """
    pts = cart_to_hom(pts)
    pts = pts @ pose.T
    pts = hom_to_cart(pts)
    return pts


@dispatch(torch.Tensor, torch.Tensor)
def canonical_to_camera(pts, pose):
    pts = cart_to_hom(pts)
    pts = pts @ pose.transpose(-1, -2)
    pts = hom_to_cart(pts)
    return pts


transform_points = canonical_to_camera

def camera_to_canonical(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :return:
    """
    if isinstance(pts, np.ndarray) and isinstance(pose, np.ndarray):
        pts = pts.T  # 3xN
        pts = np.vstack((pts, np.ones((1, pts.shape[1]))))  # 4XN
        p = np.linalg.inv(pose) @ pts  # 4xN
        p[0:3] /= p[3:]
        p = p[0:3]
        p = p.T
        return p
    else:
        pts = cart_to_hom(pts)
        pts = pts @ torch.inverse(pose).t()
        pts = hom_to_cart(pts)
        return pts

def xyzr_to_pose4x4(x, y, z, r):
    pose = np.eye(4)
    pose[0, 0] = np.cos(r)
    pose[0, 2] = np.sin(r)
    pose[2, 0] = -np.sin(r)
    pose[2, 2] = np.cos(r)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose

def xyzr_to_pose4x4_torch(x, y, z, r):
    if isinstance(x, torch.Tensor):
        pose = torch.eye(4, device=x.device, dtype=torch.float)
        pose[0, 0] = torch.cos(r)
        pose[0, 2] = torch.sin(r)
        pose[2, 0] = -torch.sin(r)
        pose[2, 2] = torch.cos(r)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z
        return pose
    else:
        return torch.from_numpy(xyzr_to_pose4x4_np(x, y, z, r)).float()

@dispatch(np.ndarray)
def pose4x4_to_xyzr(pose):
    x = pose[0, 3]
    y = pose[1, 3]
    z = pose[2, 3]
    cos = pose[0, 0]
    sin = pose[0, 2]
    angle = np.arctan2(sin, cos)
    return x, y, z, angle


@dispatch(torch.Tensor)
def pose4x4_to_xyzr(pose):
    x = pose[0, 3]
    y = pose[1, 3]
    z = pose[2, 3]
    cos = pose[0, 0]
    sin = pose[0, 2]
    angle = torch.atan2(sin, cos)
    return x, y, z, angle

def camera_coordinate_to_world_coordinate(pts_in_camera, cam_pose):
    """
    transform points in camera coordinate to points in world coordinate
    :param pts_in_camera: n,3
    :param cam_pose: 4,4
    :return:
    """
    if isinstance(pts_in_camera, np.ndarray):
        pts_hom = np.hstack((pts_in_camera, np.ones((pts_in_camera.shape[0], 1), dtype=np.float32)))
        pts_world = pts_hom @ cam_pose.T
    else:
        ones = torch.ones((pts_in_camera.shape[0], 1), dtype=torch.float32, device=pts_in_camera.device)
        pts_hom = torch.cat((pts_in_camera, ones), dim=1)
        cam_pose = torch.tensor(cam_pose).float().to(device=pts_in_camera.device)
        pts_world = pts_hom @ cam_pose.t()
    pts_world = pts_world[:, :3] / pts_world[:, 3:]
    return pts_world

def world_coordinate_to_camera_coordinate(pts_in_world, cam_pose):
    """
    transform points in camera coordinate to points in world coordinate
    :param pts_in_world: n,3
    :param cam_pose: 4,4
    :return:
    """
    if isinstance(pts_in_world, np.ndarray):
        cam_pose_inv = np.linalg.inv(cam_pose)
        pts_hom = np.hstack((pts_in_world, np.ones((pts_in_world.shape[0], 1), dtype=np.float32)))
        pts_cam = pts_hom @ cam_pose_inv.T
    else:
        cam_pose = cam_pose.float().to(device=pts_in_world.device)
        cam_pose_inv = torch.inverse(cam_pose)
        ones = torch.ones((pts_in_world.shape[0], 1), dtype=torch.float32, device=pts_in_world.device)
        pts_hom = torch.cat((pts_in_world, ones), dim=1)
        pts_cam = pts_hom @ cam_pose_inv.t()
    pts_cam = pts_cam[:, :3] / pts_cam[:, 3:]
    return pts_cam

def xyzr_to_pose4x4_np(x, y, z, r):
    pose = np.eye(4)
    pose[0, 0] = np.cos(r)
    pose[0, 2] = np.sin(r)
    pose[2, 0] = -np.sin(r)
    pose[2, 2] = np.cos(r)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose

def canonical_to_camera_np(pts, pose, calib=None):
    """
    :param pts: Nx3
    :param pose: 4x4
    :param calib: KITTICalib
    :return:
    """
    pts = pts.T  # 3xN
    pts = np.vstack((pts, np.ones((1, pts.shape[1]))))  # 4XN
    p = pose @ pts  # 4xN
    if calib is None:
        p[0:3] /= p[3:]
        p = p[0:3]
    else:
        p = calib.P2 @ p
        p[0:2] /= p[2:]
        p = p[0:2]
    p = p.T
    return p

def rotx_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([ones, zeros, zeros,
                    zeros, c, -s,
                    zeros, s, c])
    return rot.reshape((-1, 3, 3))

def roty_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    # TODO validate
    rot = np.stack([c, zeros, s,
                    zeros, ones, zeros,
                    -s, zeros, c])
    return rot.reshape((-1, 3, 3))

def roty_torch(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = torch.tensor([a])
    # if a.shape[-1] != 1:
    #     a = a[..., None]
    a = a.float()
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(a)
    c = torch.cos(a)
    s = torch.sin(a)
    # TODO validate
    rot = torch.stack([c, zeros, s,
                       zeros, ones, zeros,
                       -s, zeros, c], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)

def rotz_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([c, -s, zeros,
                    s, c, zeros,
                    zeros, zeros, ones])
    return rot.reshape((-1, 3, 3))

def rotz_torch(a):
    if isinstance(a, (int, float)):
        a = torch.tensor([a])
    if a.shape[-1] != 1:
        a = a[..., None]
    a = a.float()
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(a)
    c = torch.cos(a)
    s = torch.sin(a)
    rot = torch.stack([c, -s, zeros,
                       s, c, zeros,
                       zeros, zeros, ones], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)

def rotx(t):
    """
    Rotation along the x-axis.
    :param t: tensor of (N, 1) or (N), or float, or int
              angle
    :return: tensor of (N, 3, 3)
             rotation matrix
    """
    if isinstance(t, (int, float)):
        t = torch.tensor([t])
    if t.shape[-1] != 1:
        t = t[..., None]
    t = t.type(torch.float)
    ones = torch.ones_like(t)
    zeros = torch.zeros_like(t)
    c = torch.cos(t)
    s = torch.sin(t)
    rot = torch.stack([ones, zeros, zeros,
                       zeros, c, -s,
                       zeros, s, c], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)

def matrix_3x4_to_4x4(a):
    if len(a.shape) == 2:
        assert a.shape == (3, 4)
    else:
        assert len(a.shape) == 3
        assert a.shape[1:] == (3, 4)
    if isinstance(a, np.ndarray):
        if a.ndim == 2:
            ones = np.array([[0, 0, 0, 1]])
            return np.vstack((a, ones))
        else:
            ones = np.array([[0, 0, 0, 1]])[None].repeat(a.shape[0], axis=0)
            return np.concatenate((a, ones), axis=1)
    else:
        ones = torch.tensor([[0, 0, 0, 1]]).float().to(device=a.device)
        if a.ndim == 3:
            ones = ones[None].repeat(a.shape[0], 1, 1)
            ret = torch.cat((a, ones), dim=1)
        else:
            ret = torch.cat((a, ones), dim=0)
        return ret

def matrix_3x3_to_4x4(a):
    assert a.shape == (3, 3)
    if isinstance(a, np.ndarray):
        ret = np.eye(4)
    else:
        ret = torch.eye(4).float().to(a.device)
    ret[:3, :3] = a
    return ret

def radian_to_negpi_to_pi(x):
    if isinstance(x, torch.Tensor):
        return x - 2 * torch.floor((x / np.pi + 1) / 2) * np.pi
    else:
        return x - 2 * np.floor((x / np.pi + 1) / 2) * np.pi

def filter_bbox_3d(bbox_3d, point):
    """
    @param bbox_3d: corners (8,3)
    @param point: (?,3)
    @return:
    """
    v45 = bbox_3d[5] - bbox_3d[4]
    v40 = bbox_3d[0] - bbox_3d[4]
    v47 = bbox_3d[7] - bbox_3d[4]
    # point -= bbox_3d[4]
    point = point - bbox_3d[4]
    m0 = torch.matmul(point, v45)
    m1 = torch.matmul(point, v40)
    m2 = torch.matmul(point, v47)

    cs = []
    for m, v in zip([m0, m1, m2], [v45, v40, v47]):
        c0 = 0 < m
        c1 = m < torch.matmul(v, v)
        c = c0 & c1
        cs.append(c)
    cs = cs[0] & cs[1] & cs[2]
    passed_inds = torch.nonzero(cs).squeeze(1)
    num_passed = torch.sum(cs)
    return num_passed, passed_inds, cs

def triangulate_static_points(projmat1, projmat2, projpoints1, projpoints2) -> np.ndarray:
    """
    triangulate points python version
    :param projmat1: 3x4
    :param projmat2: 3x4
    :param projpoints1: 2xn
    :param projpoints2: 2xn
    :return: points4D: 4xn
    """
    num_points = projpoints1.shape[1]
    assert projpoints2.shape[1] == num_points
    points4D = []
    for i in range(num_points):
        x, y = projpoints1[:, i]
        xp, yp = projpoints2[:, i]
        A = np.stack([x * projmat1[2] - projmat1[0],
                      y * projmat1[2] - projmat1[1],
                      xp * projmat2[2] - projmat2[0],
                      yp * projmat2[2] - projmat2[1]], axis=0)
        U, S, V = np.linalg.svd(A)
        points4D.append(V[3, 0:4])
    points4D = np.stack(points4D).T
    return points4D


def subsample_mask_by_grid(pc_rect):
    def filter_mask(pc_rect):
        """Return index of points that lies within the region defined below."""
        valid_inds = (pc_rect[:, 2] < 80) * \
                     (pc_rect[:, 2] > 1) * \
                     (pc_rect[:, 0] < 40) * \
                     (pc_rect[:, 0] >= -40) * \
                     (pc_rect[:, 1] < 2.5) * \
                     (pc_rect[:, 1] >= -1)
        return valid_inds

    GRID_SIZE = 0.1
    index_field_sample = np.full(
        (35, int(80 / 0.1), int(80 / 0.1)), -1, dtype=np.int32)

    N = pc_rect.shape[0]
    perm = np.random.permutation(pc_rect.shape[0])
    pc_rect = pc_rect[perm]

    range_filter = filter_mask(pc_rect)
    pc_rect = pc_rect[range_filter]

    pc_rect_quantized = np.floor(pc_rect[:, :3] / GRID_SIZE).astype(np.int32)
    pc_rect_quantized[:, 0] = pc_rect_quantized[:, 0] \
                              + int(80 / GRID_SIZE / 2)
    pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] + int(1 / GRID_SIZE)

    index_field = index_field_sample.copy()

    index_field[pc_rect_quantized[:, 1],
                pc_rect_quantized[:, 2], pc_rect_quantized[:, 0]] = np.arange(pc_rect.shape[0])
    mask = np.zeros(perm.shape, dtype=np.bool)
    mask[perm[range_filter][index_field[index_field >= 0]]] = 1
    return mask


def img_to_rect(fu, fv, cu, cv, u, v, depth_rect):
    """
    :param u: (N)
    :param v: (N)
    :param depth_rect: (N)
    :return: pts_rect:(N, 3)
    """
    # check_type(u)
    # check_type(v)

    if isinstance(depth_rect, np.ndarray):
        x = ((u - cu) * depth_rect) / fu
        y = ((v - cv) * depth_rect) / fv
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    else:
        x = ((u.float() - cu) * depth_rect) / fu
        y = ((v.float() - cv) * depth_rect) / fv
        pts_rect = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1)
    # x = ((u - cu) * depth_rect) / fu
    # y = ((v - cv) * depth_rect) / fv
    # pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    return pts_rect


def depth_to_rect(fu, fv, cu, cv, depth_map, ray_mode=False, select_coords=None):
    """

    :param fu:
    :param fv:
    :param cu:
    :param cv:
    :param depth_map:
    :param ray_mode: whether values in depth_map are Z or norm
    :return:
    """
    if len(depth_map.shape) == 2:
        if isinstance(depth_map, np.ndarray):
            x_range = np.arange(0, depth_map.shape[1])
            y_range = np.arange(0, depth_map.shape[0])
            x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        else:
            x_range = torch.arange(0, depth_map.shape[1]).to(device=depth_map.device)
            y_range = torch.arange(0, depth_map.shape[0]).to(device=depth_map.device)
            y_idxs, x_idxs = torch.meshgrid(y_range, x_range, indexing='ij')
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
    else:
        x_idxs = select_coords[:, 1]
        y_idxs = select_coords[:, 0]
        depth = depth_map
    if ray_mode is True:
        if isinstance(depth, torch.Tensor):
            depth = depth / (((x_idxs.float() - cu.float()) / fu.float()) ** 2 + (
                    (y_idxs.float() - cv.float()) / fv.float()) ** 2 + 1) ** 0.5
        else:
            depth = depth / (((x_idxs - cu) / fu) ** 2 + (
                    (y_idxs - cv) / fv) ** 2 + 1) ** 0.5
    pts_rect = img_to_rect(fu, fv, cu, cv, x_idxs, y_idxs, depth)
    # if ray_mode is True:
    #     todo check this
    # pts_rect[:, 2] = (pts_rect[:, 2] ** 2 - pts_rect[:, 0] ** 2 - pts_rect[:, 1] ** 2) ** 0.5
    return pts_rect


def depth_norm_to_depth_z(K, depth_map):
    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if isinstance(depth_map, np.ndarray):
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
    else:
        x_range = torch.arange(0, depth_map.shape[1]).to(device=depth_map.device)
        y_range = torch.arange(0, depth_map.shape[0]).to(device=depth_map.device)
        y_idxs, x_idxs = torch.meshgrid(y_range, x_range, indexing='ij')
    x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
    depth = depth_map[y_idxs, x_idxs]

    if isinstance(depth, torch.Tensor):
        depth = depth / (((x_idxs.float() - cu.float()) / fu.float()) ** 2 + (
                (y_idxs.float() - cv.float()) / fv.float()) ** 2 + 1) ** 0.5
    else:
        depth = depth / (((x_idxs - cu) / fu) ** 2 + ((y_idxs - cv) / fv) ** 2 + 1) ** 0.5
    return depth.reshape(depth_map.shape)


@dispatch(float, float, float, float, np.ndarray)
def rect_to_img(fu, fv, cu, cv, pts_rect):
    K = np.array([[fu, 0, cu],
                  [0, fv, cv],
                  [0, 0, 1]])
    pts_2d_hom = pts_rect @ K.T
    pts_img = hom_to_cart(pts_2d_hom)
    return pts_img


@dispatch(float, float, float, float, torch.Tensor)
def rect_to_img(fu, fv, cu, cv, pts_rect):
    device = pts_rect.device
    P2 = torch.tensor([[fu, 0, cu],
                       [0, fv, cv],
                       [0, 0, 1]], dtype=torch.float, device=device)
    pts_2d_hom = pts_rect @ P2.t()
    pts_img = hom_to_cart(pts_2d_hom)
    return pts_img


@dispatch(np.ndarray, np.ndarray)
def rect_to_img(K, pts_rect):
    pts_2d_hom = pts_rect @ K.T
    pts_img = hom_to_cart(pts_2d_hom)
    return pts_img


@dispatch(torch.Tensor, torch.Tensor)
def rect_to_img(K, pts_rect):
    pts_2d_hom = pts_rect @ K.t()
    pts_img = hom_to_cart(pts_2d_hom)
    return pts_img


def backproject_flow3d_torch(flow2d, depth0, depth1, intrinsics, campose0, campose1):
    """ compute 3D flow from 2D flow + depth change """
    # raise NotImplementedError()

    ht, wd = flow2d.shape[0:2]

    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    y0, x0 = torch.meshgrid(
        torch.arange(ht).to(depth0.device).float(),
        torch.arange(wd).to(depth0.device).float())

    x1 = x0 + flow2d[..., 0]
    y1 = y0 + flow2d[..., 1]

    X0 = depth0 * ((x0 - cx) / fx)
    Y0 = depth0 * ((y0 - cy) / fy)
    Z0 = depth0

    # X1 = depth1 * ((x1 - cx) / fx)
    # Y1 = depth1 * ((y1 - cy) / fy)
    # Z1 = depth1

    grid = torch.stack([x1, y1], dim=-1)[None]
    grid[:, :, :, 0] = grid[:, :, :, 0] / (wd - 1)
    grid[:, :, :, 1] = grid[:, :, :, 1] / (ht - 1)
    grid = grid * 2 - 1
    depth1_interp = torch.nn.functional.grid_sample(
        depth1[None, None],
        grid,
        mode='bilinear'
    )[0, 0]

    X1 = depth1_interp * ((x1 - cx) / fx)
    Y1 = depth1_interp * ((y1 - cy) / fy)
    Z1 = depth1_interp  # todo: or interpolated depth?

    pts0_cam = torch.stack([X0, Y0, Z0], dim=-1)
    pts1_cam = torch.stack([X1, Y1, Z1], dim=-1)

    # pts0_world = camera_coordinate_to_world_coordinate(pts0_cam, campose0)
    # pts1_world = camera_coordinate_to_world_coordinate(pts1_cam, campose1)

    # flow3d = torch.stack([X1 - X0, Y1 - Y0, Z1 - Z0], dim=-1)
    flow3d = pts1_cam - pts0_cam
    return flow3d, pts0_cam, pts1_cam


def backproject_flow3d_np(flow2d, depth0, depth1, intrinsics0, intrinsics1, campose0, campose1, occlusion,
                          max_norm=0.5, interpolation_mode='nearest'):
    """
    compute 3D flow from 2D flow + depth change
    :param flow2d:
    :param depth0:
    :param depth1:
    :param intrinsics0:
    :param intrinsics1:
    :param campose0:
    :param campose1:
    :param occlusion: 0,255
    :param max_norm:
    :return:
    """

    ht, wd = flow2d.shape[0:2]

    fx0, fy0, cx0, cy0 = intrinsics0[0, 0], intrinsics0[1, 1], intrinsics0[0, 2], intrinsics0[1, 2]
    fx1, fy1, cx1, cy1 = intrinsics1[0, 0], intrinsics1[1, 1], intrinsics1[0, 2], intrinsics1[1, 2]

    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + flow2d[..., 0]
    y1 = y0 + flow2d[..., 1]

    X0 = depth0 * ((x0 - cx0) / fx0)
    Y0 = depth0 * ((y0 - cy0) / fy0)
    Z0 = depth0

    # bilinear sample
    grid = torch.from_numpy(np.stack([x1, y1], axis=-1)[None]).float()
    grid[:, :, :, 0] = grid[:, :, :, 0] / (wd - 1)
    grid[:, :, :, 1] = grid[:, :, :, 1] / (ht - 1)
    grid = grid * 2 - 1
    depth1_interp = torch.nn.functional.grid_sample(
        torch.from_numpy(depth1)[None, None],
        grid,
        mode=interpolation_mode
    )[0, 0].numpy()

    X1 = depth1_interp * ((x1 - cx1) / fx1)
    Y1 = depth1_interp * ((y1 - cy1) / fy1)
    Z1 = depth1_interp  # todo: or interpolated depth?

    pts0_cam = np.stack([X0, Y0, Z0], axis=-1).reshape(-1, 3)
    pts1_cam = np.stack([X1, Y1, Z1], axis=-1).reshape(-1, 3)
    posenorm = np.linalg.inv(campose0)
    campose0, campose1 = posenorm @ campose0, posenorm @ campose1
    pts0_world = camera_coordinate_to_world_coordinate(pts0_cam, campose0)
    pts1_world = camera_coordinate_to_world_coordinate(pts1_cam, campose1)

    # flow3d = torch.stack([X1 - X0, Y1 - Y0, Z1 - Z0], dim=-1)
    flow3d = pts1_world - pts0_world
    # with Vis3D(
    #         xyz_pattern=('x', '-y', '-z'),
    #         out_folder="dbg"
    # ) as vis3d:
    #     vis3d.set_scene_id(0)
    #     vis3d.add_point_cloud(pts0_world)
    #     vis3d.add_point_cloud(pts1_world)
    #     print()
    flow3d = flow3d.reshape(ht, wd, 3)
    norm = np.linalg.norm(flow3d, axis=-1)
    ncc = occlusion != 255
    flow3dncc = flow3d * ncc[:, :, None]
    normncc = norm * ncc
    # plt.imshow(normncc, 'jet')
    # plt.show()
    normmask = normncc < max_norm
    flow3dncc = flow3dncc * normmask[:, :, None]
    return flow3dncc, normmask


def _sample_at_integer_locs(input_feats, index_tensor):
    assert input_feats.ndimension() == 5, 'input_feats should be of shape [B,F,D,H,W]'
    assert index_tensor.ndimension() == 4, 'index_tensor should be of shape [B,H,W,3]'
    # first sample pixel locations using nearest neighbour interpolation
    batch_size, num_chans, num_d, height, width = input_feats.shape
    grid_height, grid_width = index_tensor.shape[1], index_tensor.shape[2]

    xy_grid = index_tensor[..., 0:2]
    xy_grid[..., 0] = xy_grid[..., 0] - ((width - 1.0) / 2.0)
    xy_grid[..., 0] = xy_grid[..., 0] / ((width - 1.0) / 2.0)
    xy_grid[..., 1] = xy_grid[..., 1] - ((height - 1.0) / 2.0)
    xy_grid[..., 1] = xy_grid[..., 1] / ((height - 1.0) / 2.0)
    xy_grid = torch.clamp(xy_grid, min=-1.0, max=1.0)
    sampled_in_2d = F.grid_sample(input=input_feats.view(batch_size, num_chans * num_d, height, width),
                                  grid=xy_grid, mode='nearest', align_corners=False).view(batch_size, num_chans, num_d,
                                                                                          grid_height,
                                                                                          grid_width)
    z_grid = index_tensor[..., 2].view(batch_size, 1, 1, grid_height, grid_width)
    z_grid = z_grid.long().clamp(min=0, max=num_d - 1)
    z_grid = z_grid.expand(batch_size, num_chans, 1, grid_height, grid_width)
    sampled_in_3d = sampled_in_2d.gather(2, z_grid).squeeze(2)
    return sampled_in_3d


def trilinear_interpolation(input_feats, sampling_grid):
    """
    interploate value in 3D volume
    :param input_feats: [B,C,D,H,W]
    :param sampling_grid: [B,H,W,3] unscaled coordinates
    :return:
    """
    assert input_feats.ndimension() == 5, 'input_feats should be of shape [B,F,D,H,W]'
    assert sampling_grid.ndimension() == 4, 'sampling_grid should be of shape [B,H,W,3]'
    batch_size, num_chans, num_d, height, width = input_feats.shape
    grid_height, grid_width = sampling_grid.shape[1], sampling_grid.shape[2]
    # make sure sampling grid lies between -1, 1
    sampling_grid[..., 0] = 2 * sampling_grid[..., 0] / (num_d - 1) - 1
    sampling_grid[..., 1] = 2 * sampling_grid[..., 1] / (height - 1) - 1
    sampling_grid[..., 2] = 2 * sampling_grid[..., 2] / (width - 1) - 1
    sampling_grid = torch.clamp(sampling_grid, min=-1.0, max=1.0)
    # map to 0,1
    sampling_grid = (sampling_grid + 1) / 2.0
    # Scale grid to floating point pixel locations
    scaling_factor = torch.FloatTensor([width - 1.0, height - 1.0, num_d - 1.0]).to(input_feats.device).view(1, 1,
                                                                                                             1, 3)
    sampling_grid = scaling_factor * sampling_grid
    # Now sampling grid is between [0, w-1; 0,h-1; 0,d-1]
    x, y, z = torch.split(sampling_grid, split_size_or_sections=1, dim=3)
    x_0, y_0, z_0 = torch.split(sampling_grid.floor(), split_size_or_sections=1, dim=3)
    x_1, y_1, z_1 = x_0 + 1.0, y_0 + 1.0, z_0 + 1.0
    u, v, w = x - x_0, y - y_0, z - z_0
    u, v, w = map(lambda x: x.view(batch_size, 1, grid_height, grid_width).expand(
        batch_size, num_chans, grid_height, grid_width), [u, v, w])
    c_000 = _sample_at_integer_locs(input_feats, torch.cat([x_0, y_0, z_0], dim=3))
    c_001 = _sample_at_integer_locs(input_feats, torch.cat([x_0, y_0, z_1], dim=3))
    c_010 = _sample_at_integer_locs(input_feats, torch.cat([x_0, y_1, z_0], dim=3))
    c_011 = _sample_at_integer_locs(input_feats, torch.cat([x_0, y_1, z_1], dim=3))
    c_100 = _sample_at_integer_locs(input_feats, torch.cat([x_1, y_0, z_0], dim=3))
    c_101 = _sample_at_integer_locs(input_feats, torch.cat([x_1, y_0, z_1], dim=3))
    c_110 = _sample_at_integer_locs(input_feats, torch.cat([x_1, y_1, z_0], dim=3))
    c_111 = _sample_at_integer_locs(input_feats, torch.cat([x_1, y_1, z_1], dim=3))
    c_xyz = (1.0 - u) * (1.0 - v) * (1.0 - w) * c_000 + \
            (1.0 - u) * (1.0 - v) * w * c_001 + \
            (1.0 - u) * v * (1.0 - w) * c_010 + \
            (1.0 - u) * v * w * c_011 + \
            u * (1.0 - v) * (1.0 - w) * c_100 + \
            u * (1.0 - v) * w * c_101 + \
            u * v * (1.0 - w) * c_110 + \
            u * v * w * c_111
    return c_xyz


def create_center_radius(center=np.array([0, 0, 0]), dist=5., angle_z=30, nrad=180, start=0., endpoint=True,
                         end=2 * np.pi):
    RTs = []
    center = np.array(center).reshape(3, 1)
    thetas = np.linspace(start, end, nrad, endpoint=endpoint)
    angle_z = np.deg2rad(angle_z)
    radius = dist * np.cos(angle_z)
    height = dist * np.sin(angle_z)
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        center_ = np.array([radius * ct, radius * st, height]).reshape(3, 1)
        center_[0] += center[0, 0]
        center_[1] += center[1, 0]
        R = np.array([
            [-st, ct, 0],
            [0, 0, -1],
            [-ct, -st, 0]
        ])
        Rotx = cv2.Rodrigues(angle_z * np.array([1., 0., 0.]))[0]
        R = Rotx @ R
        T = - R @ center_
        center_ = - R.T @ T
        RT = np.hstack([R, T])
        RTs.append(RT)
    return np.stack(RTs)

    vol_bnds = np.zeros((3, 2))
    # obj_poses = obj_poses @ np.linalg.inv(obj_poses[index])
    for depth_im, op in safe_zip(depths, obj_poses):
        cam_pose = np.linalg.inv(op)
        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    vol_bnds[:, 0] = np.floor(vol_bnds[:, 0] / voxel_length) * voxel_length
    vol_bnds[:, 1] = np.ceil(vol_bnds[:, 1] / voxel_length) * voxel_length
    return vol_bnds


def open3d_icp_api(pts0, pts1, thresh, init_Rt=np.eye(4), return_tsfm_only=True, use_cuda=False, voxel_size=-1):
    """
    R*pts0+t=pts1
    :param pts0: nx3
    :param pts1: mx3
    :param thresh: float
    :param init_Rt: 4x4
    :return:
    """
    import open3d as o3d
    if not use_cuda:
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(pts0.copy())
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts1.copy())
        if version.parse(o3d.__version__) < version.parse('0.10.0'):
            result = o3d.registration.registration_icp(
                pcd0, pcd1, thresh, init_Rt)
        else:
            result = o3d.pipelines.registration.registration_icp(
                pcd0, pcd1, thresh, init_Rt)
        del pcd0, pcd1
        if return_tsfm_only:
            return result.transformation
        else:
            return result
    else:
        import open3d.cuda.pybind.t.pipelines.registration as treg
        pcd0 = o3d.t.geometry.PointCloud(o3d.cuda.pybind.core.Tensor(pts0.astype(np.float32)))
        # pcd0.point.positions =
        pcd1 = o3d.t.geometry.PointCloud(o3d.cuda.pybind.core.Tensor(pts1.astype(np.float32)))
        # pcd1.point.positions = o3d.core.Tensor(pts1.astype(np.float32))
        result = treg.icp(pcd0, pcd1, thresh,
                          init_Rt,
                          # estimation,
                          # voxel_size=voxel_size,
                          )
        del pcd0, pcd1
        if return_tsfm_only:
            return result.transformation.numpy()
        else:
            return result


def open3d_ransac_api(pts0, pts1, thresh):
    """
    R*pts0+t=pts1
    :param pts0: nx3
    :param pts1: mx3
    :param thresh: float
    :param init_Rt: 4x4
    :return:
    """
    import open3d as o3d
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts0)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    corres = np.arange(pts0.shape[0])[:, None].repeat(2, axis=1)
    corres = o3d.utility.Vector2iVector(corres)
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd0, pcd1, corres, thresh)
    return result.transformation


def open3d_colored_icp_api(src_pc, tgt_pc, src_color, tgt_color, init_tsfm=np.eye(4)):
    import open3d as o3d

    if isinstance(src_pc, trimesh.Trimesh):
        src_pc = src_pc.vertices
    if isinstance(src_pc, torch.Tensor):
        src_pc = src_pc.cpu().numpy()
    if isinstance(tgt_pc, trimesh.Trimesh):
        tgt_pc = tgt_pc.vertices
    if isinstance(tgt_pc, torch.Tensor):
        tgt_pc = tgt_pc.cpu().numpy()
    # if normalize_scale:
    #     scaling = 1.0 / (src_pc.max(0) - src_pc.min(0)).max()
    #     src_pc = src_pc * scaling
    # scaling = 1.0 / (tgt_pc.max(0) - tgt_pc.min(0)).max()
    # tgt_pc = tgt_pc * scaling
    # if normalize_position:
    #     src_pc = src_pc - src_pc.min(0)
    #     tgt_pc = tgt_pc - tgt_pc.min(0)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src_pc.copy())
    source.colors = o3d.utility.Vector3dVector(src_color.copy())

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(tgt_pc.copy())
    target.colors = o3d.utility.Vector3dVector(tgt_color.copy())

    # voxel_radius = [0.04, 0.02, 0.01]
    # voxel_radius = [0.02, 0.01]
    voxel_radius = [0.01]
    # voxel_radius = [0.08, 0.04, 0.02]
    # voxel_radius = [0.16, 0.08, 0.04]
    # max_iter = [50, 30, 14]
    max_iter = [14]
    current_transformation = init_tsfm
    print("3. Colored point cloud registration")
    results_icp = []
    for scale in range(len(voxel_radius)):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_icp.transformation
        results_icp.append(result_icp)
        # print(result_icp)
    return results_icp


def cpa_pytorch3d_api(pts0, pts1, estimate_scale=False, use_gpu=True):
    """
    R*pts0+t=pts1
    :param pts0: nx3
    :param pts1: nx3
    :return: 4x4
    """
    from pytorch3d.ops import corresponding_points_alignment
    pts0 = torch.from_numpy(pts0).float()
    pts1 = torch.from_numpy(pts1).float()
    if use_gpu:
        pts0 = pts0.cuda()
        pts1 = pts1.cuda()
    cpa_res = corresponding_points_alignment(pts0[None], pts1[None], estimate_scale=estimate_scale)
    R = cpa_res.R[0].T
    t = cpa_res.T[0]
    Rt = torch.cat([R, t.reshape(3, 1)], dim=1)
    pose = matrix_3x4_to_4x4(Rt)
    return pose.cpu().numpy()


@dispatch(np.ndarray)
def interp_pose(poses):
    """

    :param poses: np.ndarray N,4,4
    :return:
    """
    N = len(poses)
    nN = N * 2 - 1
    newposes = np.zeros([nN, 4, 4])
    newposes[::2, :, :] = poses
    a = poses[:-1]
    b = poses[1:]
    aa = se3_log_map(torch.from_numpy(a.transpose(0, 2, 1)))
    bb = se3_log_map(torch.from_numpy(b.transpose(0, 2, 1)))
    cc = (aa + bb) / 2
    c = se3_exp_map(cc).numpy().transpose(0, 2, 1)
    newposes[1::2, :, :] = c
    return newposes


def compose_pair(pose_a, pose_b):
    # pose_new(x) = pose_b o pose_a(x)
    R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
    R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
    R_new = R_b @ R_a
    t_new = (R_b @ t_a + t_b)[..., 0]
    pose_new = torch.cat([R_new, t_new[..., None]], dim=-1)
    pose_new = matrix_3x4_to_4x4(pose_new)
    return pose_new


def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    if R1.ndim == 2 and R2.ndim == 2:
        R_diff = R1[:3, :3] @ R2[:3, :3].T
        trace = R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]
    else:
        R_diff = R1[..., :3, :3] @ R2.transpose(-2, -1)[..., :3, :3]
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()  # numerical stability near -1/+1
    return angle


def pose_distance(pred, gt, eps=1e-7, align=False):
    if pred.numel() == 0 or gt.numel() == 0:
        return torch.empty([0]), torch.empty([0])
    if pred.ndim == 2 and gt.ndim == 2:
        pred = pred[None]
        gt = gt[None]
    if align:
        gt = gt @ gt[0].inverse()[None]
        pred = pred @ pred[0].inverse()[None]
    R_error = rotation_distance(pred, gt, eps)
    t_error = (pred[..., :3, 3] - gt[..., :3, 3]).norm(dim=-1)
    return R_error, t_error


def project_to_img(pts_cam, K, shape):
    tmp = torch.zeros(shape)
    K = K.cpu()
    pts_img = rect_to_img(K[0, 0], K[1, 1], K[0, 2], K[1, 2], pts_cam.cpu())
    tmp[pts_img[:, 1].long(), pts_img[:, 0].long()] = 1
    return tmp


def chamfer_distance_and_fscore(pts0, pts1, use_gpu=True, threshold=0.05):
    if use_gpu:
        from dmv.utils.chamfer3D import dist_chamfer_3D
        chamLoss = dist_chamfer_3D.chamfer_3DDist()
        points1 = to_tensor(pts0, device='cuda').float()[None]
        points2 = to_tensor(pts1, device='cuda').cuda().float()[None]
        # points1 = torch.rand(32, 1000, 3).cuda()
        # points2 = torch.rand(32, 2000, 3, requires_grad=True).cuda()
        dist1, dist2, idx1, idx2 = chamLoss(points1, points2)
        dist1 = dist1 ** 0.5
        dist2 = dist2 ** 0.5
        loss = dist1.mean() + dist2.mean()
        dist2_threshed = dist2 < threshold
        dist1_threshed = dist1 < threshold
        if dist2_threshed.sum() == 0 or dist1_threshed.sum() == 0:
            fscore = torch.zeros(1, device='cuda')
        else:
            precision = (dist2 < threshold).float().mean()
            recal = (dist1 < threshold).float().mean()
            fscore = 2 * precision * recal / (precision + recal)
        return loss.item(), fscore.item()
    else:
        raise NotImplementedError()
        pts0 = to_tensor(pts0)
        pts1 = to_tensor(pts1)

        def square_distance(src, dst):
            return torch.sum((src[:, None, :] - dst[None, :, :]) ** 2, dim=-1)

        dist_src = torch.min(square_distance(pts0, pts1), dim=-1)
        dist_ref = torch.min(square_distance(pts1, pts0), dim=-1)
        chamfer_dist = torch.mean(dist_src[0]) + torch.mean(dist_ref[0])
        if color0 is not None or color1 is not None:
            raise NotImplementedError()
        return chamfer_dist.item()


def open3d_plane_segment_api(pts, distance_threshold, ransac_n=3, num_iterations=1000):
    import open3d as o3d

    pts = to_array(pts)
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts)
    plane_model, inliers = pcd0.segment_plane(distance_threshold,
                                              ransac_n=ransac_n,
                                              num_iterations=num_iterations)
    return plane_model, inliers


def point_plane_distance_api(pts, plane_model):
    a, b, c, d = plane_model.tolist()
    if isinstance(pts, torch.Tensor):
        dists = (a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d).abs() / ((a * a + b * b + c * c) ** 0.5)
    else:
        dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / ((a * a + b * b + c * c) ** 0.5)
    return dists


def se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4):
    return pytorch3d.transforms.se3.se3_exp_map(log_transform, eps)


def se3_log_map(transform: torch.Tensor, eps: float = 1e-4, cos_bound: float = 1e-4, backend=None,
                test_acc=True):
    if backend is None:
        loguru.logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        loguru.logger.warning("!!!!se3_log_map backend is None!!!!")
        loguru.logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        backend = 'pytorch3d'
    if backend == 'pytorch3d':
        dof6 = pytorch3d.transforms.se3.se3_log_map(transform, eps, cos_bound)
    elif backend == 'opencv':
        from pytorch3d.transforms.se3 import _se3_V_matrix, _get_se3_V_input
        # from pytorch3d.common.compat import solve
        log_rotation = []
        for tsfm in transform:
            cv2_rot = -cv2.Rodrigues(to_array(tsfm[:3, :3]))[0]
            log_rotation.append(torch.from_numpy(cv2_rot.reshape(-1)).to(transform.device).float())
        log_rotation = torch.stack(log_rotation, dim=0)
        T = transform[:, 3, :3]
        V = _se3_V_matrix(*_get_se3_V_input(log_rotation), eps=eps)
        log_translation = torch.linalg.solve(V, T[:, :, None])[:, :, 0]
        dof6 = torch.cat((log_translation, log_rotation), dim=1)
    else:
        raise NotImplementedError()
    if test_acc:
        err = (se3_exp_map(dof6) - transform).abs().max()
        if err > 0.1:
            raise RuntimeError()
    return dof6


def ray_box_intersection(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected

    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary

    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified [boxes,3]
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified [boxes,3]

    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = torch.ones_like(ray_o) * -1.  # tf.constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = torch.ones_like(ray_o)  # tf.constant([1., 1., 1.])
    inv_d = torch.reciprocal(ray_d)

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d
    t0 = torch.minimum(t_min, t_max)
    t1 = torch.maximum(t_min, t_max)

    t_near = torch.maximum(torch.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = torch.minimum(torch.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])

    # Check if rays are inside boxes
    intersection_map = torch.nonzero(t_far > t_near)
    # Check that boxes are in front of the ray origin
    positive_far = torch.nonzero(t_far[intersection_map[:, 0], intersection_map[:, 1]] > 0)
    # positive_far = torch.nonzero(tf.gather_nd(t_far, intersection_map) > 0)
    # intersection_map = tf.gather_nd(intersection_map, positive_far)
    intersection_map = intersection_map[positive_far[:, 0]]

    if intersection_map.shape[0] != 0:
        z_ray_in = t_near[intersection_map[:, 0], intersection_map[:, 1]]
        z_ray_out = t_far[intersection_map[:, 0], intersection_map[:, 1]]
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


def pts_to_box(pts, vsize, radius, dbg=False):
    occ_map = point_cloud_to_volume(pts, vsize, radius).sum(-1) > 0
    retval, labels, stats, cent = cv2.connectedComponentsWithStats(occ_map.astype(np.uint8))

    maxcomp = np.argmax(stats[1:, 4]) + 1
    targetpts = np.argwhere(labels == maxcomp)
    targetpts[..., [0, 1]] = targetpts[..., [1, 0]]

    rect = cv2.minAreaRect(targetpts)

    if dbg:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        tmp = (255 * occ_map).copy().astype(np.uint8)
        cv2.drawContours(tmp, [box], 0, 255, 1)
        plt.imshow(tmp)
        plt.show()
        print()
    voxel = 2 * radius / float(vsize)
    (x, y), (width, height), theta = rect
    x = x * voxel - radius
    y = y * voxel - radius
    width = width * voxel
    height = height * voxel
    return (x, y), (width, height), theta


def euler_to_pose(euler):
    """

    :param euler: ...,tx,ty,tz,roll pitch yaw
    :return:
    """
    if len(euler.shape) == 1:
        R = Rotation.from_euler("xyz", euler[3:]).as_matrix()
        t = euler[:3]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
    else:
        raise NotImplementedError()
    return pose


def pose_to_euler(pose):
    """

    :param pose:
    :return: tx,ty,tz,roll pitch yaw
    """
    rot = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")
    trans = pose[:3, 3]
    euler = np.concatenate([trans, rot])
    return euler


def Rt_to_pose(R, t=np.zeros(3)):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def reproj_error(K, pose, pts2d, pts3d):
    pts_cam = transform_points(pts3d, pose)
    pts_img = rect_to_img(K[0, 0], K[1, 1], K[0, 2], K[1, 2], pts_cam)
    err = np.linalg.norm(pts_img - pts2d, axis=-1).mean()
    return err


def pnp(points_3d, points_2d, K, method=cv2.SOLVEPNP_ITERATIVE, ransac=False, ransac_reproj_error=1.0,
        ransac_iterations_count=10000):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method == cv2.SOLVEPNP_EPNP:
        points_3d = np.expand_dims(points_3d, 0)
        points_2d = np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    K = K.astype(np.float64)
    if not ransac:
        _, R_exp, t = cv2.solvePnP(points_3d,
                                   points_2d,
                                   K,
                                   dist_coeffs,
                                   flags=method)
    else:
        _, R_exp, t, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            K,
            dist_coeffs,
            reprojectionError=ransac_reproj_error,
            iterationsCount=ransac_iterations_count,
            flags=cv2.SOLVEPNP_EPNP,
        )
    # , None, None, False, cv2.SOLVEPNP_UPNP)

    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    # pose=np.concatenate([R, t], axis=-1)
    pose = Rt_to_pose(R, t[:, 0])
    return pose


def query_pose_error(pose_pred, pose_gt, unit='m'):
    """
    Input:
    -----------
    pose_pred: np.array 3*4 or 4*4
    pose_gt: np.array 3*4 or 4*4
    """
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_gt.shape[0] == 4:
        pose_gt = pose_gt[:3]

    # Convert results' unit to cm
    if unit == 'm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) * 100
    elif unit == 'cm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3])
    elif unit == 'mm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) / 10
    else:
        raise NotImplementedError

    rotation_diff = np.dot(pose_pred[:, :3], pose_gt[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
    return angular_distance, translation_distance


def dummy_sdf_volume():
    sphere_radius = 0.2  # Sphere radius (in meters)
    sdf_volume_size = 1.0  # SDF volume size (in meters)
    resolution = 128  # Resolution of the SDF volume

    # Calculate the grid spacing
    grid_spacing = sdf_volume_size / resolution

    # Create an empty SDF volume grid
    sdf_volume = np.zeros((resolution, resolution, resolution), dtype=float)

    # Loop through each point in the SDF volume grid
    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                # Calculate the world coordinates of the current point
                world_x = x * grid_spacing - sdf_volume_size / 2.0
                world_y = y * grid_spacing - sdf_volume_size / 2.0
                world_z = z * grid_spacing - sdf_volume_size / 2.0

                # Calculate the distance from the current point to the sphere's center
                distance_to_center = np.sqrt(
                    (world_x ** 2) + (world_y ** 2) + (world_z ** 2)
                )

                # Calculate the SDF value for the current point
                sdf_volume[x, y, z] = distance_to_center - sphere_radius
    return sdf_volume


def calc_pose(phis, thetas, size, radius=1.2):
    import torch
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    device = torch.device('cpu')
    thetas = torch.FloatTensor(thetas).to(device)
    phis = torch.FloatTensor(phis).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        -radius * torch.cos(thetas) * torch.sin(phis),
        radius * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = normalize(centers).squeeze(0)
    up_vector = torch.FloatTensor([0, 0, 1]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = normalize(torch.cross(up_vector, forward_vector, dim=-1))
    if right_vector.pow(2).sum() < 0.01:
        right_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    return poses

_dsp_model=None

def simplify_mesh(input_mesh: trimesh):
    if input_mesh.vertices.shape[0]==0:
        return input_mesh
    global _dsp_model
    if _dsp_model is None:
        from dsp import DiffSP
        _dsp_model = DiffSP()
    verts, faces = _dsp_model.forward(torch.from_numpy(input_mesh.vertices).float().cuda(), torch.from_numpy(input_mesh.faces).int().cuda(), iter=1000000, threshold=0.0001)
    verts = verts.cpu().numpy()
    faces = faces.cpu().numpy()
    output_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return output_mesh