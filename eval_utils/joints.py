import numpy as np

from eval_utils.utils_3d import transform_points

class Joint:

    def __init__(self, joint_data:dict, method=None):

        self.valid_types = ['revolute', 'prismatic']  # Define valid types
        if joint_data['type'] not in self.valid_types:
            raise ValueError(f"Invalid type: {joint_data['type']}. Must be one of {self.valid_types}")
        self.type = str(joint_data['type'])
        self.range = np.array(joint_data['range']).astype(np.float64)
        self.axis_orig = np.array(joint_data['axis']['origin']).astype(np.float64)
        self.axis_dir = np.array(joint_data['axis']['direction']).astype(np.float64)
        
        # If ours, apply the opengl to opencv transform
        if method == 'ours':
            self.axis_orig = np.array([self.axis_orig[0], self.axis_orig[2], -self.axis_orig[1]]).astype(np.float64)
            self.axis_dir = np.array([self.axis_dir[0], self.axis_dir[2], -self.axis_dir[1]]).astype(np.float64)

    def read_ours(self, joint_data: dict):
        """Read joint data from Ours format

        Args:
            joint_data (dict): Joint data from Ours format
        """
        self.type = str(joint_data['type'])
        self.range = np.array(joint_data['range'])
        self.axis_orig = np.array(joint_data['axis']['origin']).astype(np.float64)
        self.axis_dir = np.array(joint_data['axis']['direction']).astype(np.float64)

    def apply_transform(self, transform: np.ndarray):
        """Apply transformation to the joint

        Args:
            transform (np.ndarray): Transformation matrix
        """
        self.axis_orig = transform_points(self.axis_orig[None], transform)[0]
        self.axis_dir = transform_points(self.axis_dir[None], transform)[0]

    def apply_scale(self, scale: float):
        """Apply scale to the joint

        Args:
            scale (float): Scale factor
        """
        self.axis_orig *= scale
        self.axis_dir *= scale
        if self.type == 'prismatic':
            self.range *= scale
    
    def apply_translation(self, translation: np.ndarray):
        """Apply translation to the joint

        Args:
            translation (np.ndarray): Translation vector
        """
        self.axis_orig += translation

def eval_joint(pred_joint: Joint, gt_joint: Joint):
    """Evaluate joint metrics

    Args:
        pred_joint (Joint): Predicted joint
        gt_joint (Joint): Ground truth joint
    Returns:
        dict: Joint metrics
    """
    res = {}
    
    # Compute joint axis error, compute the angle between the two axis
    axis_pred = pred_joint.axis_dir
    axis_gt = gt_joint.axis_dir
    axis_pred = axis_pred / np.linalg.norm(axis_pred)
    axis_gt = axis_gt / np.linalg.norm(axis_gt)

    axis_err = min(
        np.arccos(np.dot(axis_pred, axis_gt) / (np.linalg.norm(axis_pred) * np.linalg.norm(axis_gt))),
        np.arccos(np.dot(axis_pred, -axis_gt) / (np.linalg.norm(axis_pred) * np.linalg.norm(axis_gt)))
    )
    res['joint_axis_err'] = axis_err
    
    # Compute joint origin error
    if gt_joint.type == 'revolute':
        orig_diff = pred_joint.axis_orig - gt_joint.axis_orig
        orig_err = np.linalg.norm(np.dot(orig_diff, np.cross(axis_pred, axis_gt))) / np.linalg.norm(np.cross(axis_pred, axis_gt))
    elif gt_joint.type == 'prismatic':
        # orig_err = np.linalg.norm(pred_joint.axis_orig - gt_joint.axis_orig)
        orig_err = np.nan
    res['joint_orig_err'] = orig_err 

    return res