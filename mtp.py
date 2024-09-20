
"""
    Implementation of Multiple-Trajectory Prediction (MTP) model
    based on https://arxiv.org/pdf/1809.10732.pdf
"""


import math
import random
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as f

from nuscenes.prediction.models.backbone import calculate_backbone_feature_dim

ASV_DIM = 3


class multiple_trajectory_model(nn.Module):
    

    def __init__(self, backbone: nn.Module, num_modes: int,
                 seconds: float = 6, frequency_in_hz: float = 2,
                 n_hidden_layers: int = 4096, input_shape: Tuple[int, int, int] = (3, 500, 500)):
        super().__init__()

        self.backbone = backbone
        self.num_modes = num_modes
        backbone_feature_dim = calculate_backbone_feature_dim(backbone, input_shape)
        self.fc1 = nn.Linear(backbone_feature_dim + ASV_DIM, n_hidden_layers)
        # self.fc1 = nn.Linear(backbone_feature_dim, n_hidden_layers)
        predictions_per_mode = int(seconds*frequency_in_hz) * 2
        
        # Debug prints with explicit type checks
        size = int(num_modes * predictions_per_mode + num_modes)
        print(f'n_hidden_layers: {n_hidden_layers}, num_modes: {num_modes}, predictions_per_mode: {predictions_per_mode}')
        print(f'Calculated size for Linear layer: {size}')
        print(f'Type of calculated size: {type(size)}')


        self.fc2 = nn.Linear(n_hidden_layers, int(num_modes * predictions_per_mode + num_modes))

    def forward(self, image_tensor: torch.Tensor,
                agent_state_vector: torch.Tensor) -> torch.Tensor:
       
        backbone_features = self.backbone(image_tensor)
        features = torch.cat([backbone_features, agent_state_vector], dim=1)
        predictions = self.fc2(self.fc1(features))
        mode_probabilities = predictions[:, -self.num_modes:].clone()
        if not self.training:
            mode_probabilities = f.softmax(mode_probabilities, dim=-1)
        predictions = predictions[:, :-self.num_modes]
        return torch.cat((predictions, mode_probabilities), 1)


class multiple_trajectory_loss:
    """ Computes the loss for the MTP model. """

    def __init__(self,
                 num_modes: int,
                 regression_loss_weight: float = 1.,
                 angle_threshold_degrees: float = 5.):

        self.num_modes = num_modes
        self.num_location_coordinates_predicted = 2  
        self.regression_loss_weight = regression_loss_weight
        self.angle_threshold = angle_threshold_degrees

    def _get_trajectory_and_modes(self,
                                  model_prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mode_probabilities = model_prediction[:, -self.num_modes:].clone()

        desired_shape = (model_prediction.shape[0], self.num_modes, -1, self.num_location_coordinates_predicted)
        trajectories_no_modes = model_prediction[:, :-self.num_modes].clone().reshape(desired_shape)

        return trajectories_no_modes, mode_probabilities

    @staticmethod
    def _angle_between(ref_traj: torch.Tensor,
                       traj_to_compare: torch.Tensor) -> float:
        EPSILON = 1e-5
        if (ref_traj.ndim != 2 or traj_to_compare.ndim != 2 or
                ref_traj.shape[1] != 2 or traj_to_compare.shape[1] != 2):
            raise ValueError('Both tensors should have shapes (-1, 2).')
        if torch.isnan(traj_to_compare[-1]).any() or torch.isnan(ref_traj[-1]).any():
            return 180. - EPSILON
        traj_norms_product = float(torch.norm(ref_traj[-1]) * torch.norm(traj_to_compare[-1]))
        if math.isclose(traj_norms_product, 0):
            return 0.
        dot_product = float(ref_traj[-1].dot(traj_to_compare[-1]))
        angle = math.degrees(math.acos(max(min(dot_product / traj_norms_product, 1), -1)))

        if angle >= 180:
            return angle - EPSILON

        return angle

    @staticmethod
    def _compute_ave_l2_norms(tensor: torch.Tensor) -> float:
        l2_norms = torch.norm(tensor, p=2, dim=2)
        avg_distance = torch.mean(l2_norms)
        return avg_distance.item()

    def _compute_angles_from_ground_truth(self, target: torch.Tensor,
                                          trajectories: torch.Tensor) -> List[Tuple[float, int]]:
            :return: List of angle, index tuples.
        angles_from_ground_truth = []
        for mode, mode_trajectory in enumerate(trajectories):
            angle = self._angle_between(target[0], mode_trajectory)
            angles_from_ground_truth.append((angle, mode))
        return angles_from_ground_truth

    def _compute_best_mode(self,
                           angles_from_ground_truth: List[Tuple[float, int]],
                           target: torch.Tensor, trajectories: torch.Tensor) -> int:
        angles_from_ground_truth = sorted(angles_from_ground_truth)
        max_angle_below_thresh_idx = -1
        for angle_idx, (angle, mode) in enumerate(angles_from_ground_truth):
            if angle <= self.angle_threshold:
                max_angle_below_thresh_idx = angle_idx
            else:
                break
        if max_angle_below_thresh_idx == -1:
            best_mode = random.randint(0, self.num_modes - 1)
        else:
            distances_from_ground_truth = []

            for angle, mode in angles_from_ground_truth[:max_angle_below_thresh_idx + 1]:
                norm = self._compute_ave_l2_norms(target - trajectories[mode, :, :])
                distances_from_ground_truth.append((norm, mode))
            distances_from_ground_truth = sorted(distances_from_ground_truth)
            best_mode = distances_from_ground_truth[0][1]

        return best_mode

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_losses = torch.Tensor().requires_grad_(True).to(predictions.device)
        trajectories, modes = self._get_trajectory_and_modes(predictions)

        for batch_idx in range(predictions.shape[0]):

            angles = self._compute_angles_from_ground_truth(target=targets[batch_idx],
                                                            trajectories=trajectories[batch_idx])
            best_mode = self._compute_best_mode(angles,
                                                target=targets[batch_idx],
                                                trajectories=trajectories[batch_idx])

            best_mode_trajectory = trajectories[batch_idx, best_mode, :].unsqueeze(0)

            regression_loss = f.smooth_l1_loss(best_mode_trajectory, targets[batch_idx])

            mode_probabilities = modes[batch_idx].unsqueeze(0)
            best_mode_target = torch.tensor([best_mode], device=predictions.device)
            classification_loss = f.cross_entropy(mode_probabilities, best_mode_target)

            loss = classification_loss + self.regression_loss_weight * regression_loss

            deg = abs(math.atan( targets[batch_idx][0][-1][1]/targets[batch_idx][0][-1][0])*180/math.pi)
            deg_weight = math.exp(deg/20)
            batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)

        avg_loss = torch.mean(batch_losses)

        return avg_loss
