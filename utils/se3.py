import torch


def quaternion_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions in xyzw order to rotation matrices.
    """
    quaternion = torch.nn.functional.normalize(quaternion, p=2, dim=-1)
    x, y, z, w = torch.unbind(quaternion, dim=-1)

    two_s = 2.0
    return torch.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def quaternion_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions in wxyz order to rotation matrices.
    """
    return quaternion_xyzw_to_matrix(
        torch.cat((quaternion[..., 1:], quaternion[..., :1]), dim=-1)
    )


def position_quaternion_xyzw_to_se3(
    position: torch.Tensor,
    quaternion: torch.Tensor,
) -> torch.Tensor:
    """
    Convert position and xyzw quaternion tensors to SE(3) matrices.
    """
    return position_rotation_matrix_to_se3(
        position,
        quaternion_xyzw_to_matrix(quaternion),
    )


def position_quaternion_wxyz_to_se3(
    position: torch.Tensor,
    quaternion: torch.Tensor,
) -> torch.Tensor:
    """
    Convert position and wxyz quaternion tensors to SE(3) matrices.
    """
    return position_rotation_matrix_to_se3(
        position,
        quaternion_wxyz_to_matrix(quaternion),
    )


def pose_xyz_wxyz_to_se3(pose: torch.Tensor) -> torch.Tensor:
    """
    Convert [x, y, z, qw, qx, qy, qz] pose tensors to SE(3) matrices.
    """
    return position_quaternion_wxyz_to_se3(pose[..., :3], pose[..., 3:])


def position_rotation_matrix_to_se3(
    position: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    se3 = torch.zeros(
        *position.shape[:-1],
        4,
        4,
        dtype=position.dtype,
        device=position.device,
    )
    se3[..., :3, :3] = rotation
    se3[..., :3, 3] = position
    se3[..., 3, 3] = 1
    return se3
