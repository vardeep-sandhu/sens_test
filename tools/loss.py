import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

# from tools.voxelnet_point_sampler import VoxelFeatureAggregator


def trajectory_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the loss for the entire trajectory.
    :param predictions: Predicted poses (lookahead, 7)
                        Contains translation (x, y, z) and quaternion (qw, qx, qy, qz).
    :param targets: Ground truth poses (lookahead, 7)
                    Contains translation (x, y, z) and quaternion (qw, qx, qy, qz).
    :return: Scalar loss value
    """
    # Separate translation and rotation components
    batch_size, lookahead, _ = predictions.shape

    pred_translation = predictions[:, :, :3]  # Shape: (batch_size, lookahead, 3)
    pred_quaternion = predictions[:, :, 3:]  # Shape: (batch_size, lookahead, 4)

    target_translation = targets[:, :, :3]  # Shape: (batch_size, lookahead, 3)
    target_quaternion = targets[:, :, 3:]  # Shape: (batch_size, lookahead, 4)

    # Translation loss (MSE)
    translation_loss = F.mse_loss(pred_translation, target_translation)

    # Rotation loss (Quaternion distance)
    # Ensure quaternion normalization for robustness
    pred_quaternion = F.normalize(pred_quaternion, dim=-1)
    target_quaternion = F.normalize(target_quaternion, dim=-1)

    # Compute quaternion difference: q_diff = q1 * conjugate(q2)
    q_conjugate = torch.cat(
        [target_quaternion[:, :, :1], -target_quaternion[:, :, 1:]], dim=-1
    )
    q_diff = quaternion_multiply(
        pred_quaternion, q_conjugate
    )  # Batch quaternion difference

    # Extract the scalar part (w) of the quaternion
    w = q_diff[:, :, 0]  # Shape: (batch_size, lookahead)

    # Quaternion distance (angle difference): acos(|w|)
    rotation_loss = torch.mean(
        2 * torch.acos(torch.clamp(w.abs(), -1 + 1e-7, 1 - 1e-7))
    )

    # Combine losses (adjust weights as needed)
    total_loss = translation_loss + rotation_loss

    return total_loss


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Perform quaternion multiplication: q1 * q2.
    :param q1: First quaternion tensor of shape (..., 4)
    :param q2: Second quaternion tensor of shape (..., 4)
    :return: Resulting quaternion tensor of shape (..., 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def trajectory_loss_rotation(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Compute the loss for the entire trajectory.
    :param predictions: Predicted poses (lookahead, 12)
    :param targets: Ground truth poses (lookahead, 12)
    :return: Scalar loss value
    """
    # Separate translation and rotation components
    batch_size, lookahead, _ = predictions.shape

    pred_translation = predictions[:, :, :3]  # Shape: (lookahead, 3)
    pred_rotation = predictions[:, :, 3:].view(
        batch_size, lookahead, 3, 3
    )  # Shape: (lookahead, 3, 3)

    target_translation = targets[:, :, :3]  # Shape: (lookahead, 3)
    target_rotation = targets[:, :, 3:].view(
        batch_size, lookahead, 3, 3
    )  # Shape: (lookahead, 3, 3)

    # Translation loss (MSE)
    translation_loss = F.mse_loss(pred_translation, target_translation)

    # Rotation loss (Geodesic distance)
    identity_matrix = torch.eye(3, device=pred_rotation.device).expand(
        batch_size, lookahead, 3, 3
    )
    rotation_diff = torch.matmul(
        pred_rotation.transpose(-2, -1), target_rotation
    )  # Shape: (batch_size, lookahead, 3, 3)

    trace = torch.diagonal(rotation_diff, dim1=-2, dim2=-1).sum(
        -1
    )  # Batch-wise trace, shape: (batch_size, lookahead)
    rotation_loss = torch.mean(
        torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
    )  # Mean over batch and lookahead

    # Combine losses (you can adjust the weights if necessary)
    total_loss = translation_loss + rotation_loss

    return total_loss


def compute_ade_fde(
    predictions: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ADE and FDE for trajectory predictions.
    :param predictions: Predicted poses (lookahead, 12) or (batch_size, lookahead, 12)
    :param targets: Ground truth poses (lookahead, 12) or (batch_size, lookahead, 12)
    :return: ADE and FDE values
    """
    # Extract translation components (position) from predictions and targets
    pred_translation = predictions[..., :3]  # Shape: (..., lookahead, 3)
    target_translation = targets[..., :3]  # Shape: (..., lookahead, 3)

    # Compute displacement error at each time step
    displacement_errors = torch.norm(
        pred_translation - target_translation, dim=-1
    )  # Shape: (..., lookahead)

    # Compute ADE: Average over all time steps
    ade = displacement_errors.mean(dim=-1)  # Shape: (...) (e.g., batch-wise average)

    # Compute FDE: Error at the final time step
    fde = displacement_errors[..., -1]  # Shape: (...) (e.g., batch-wise final error)

    return ade, fde
