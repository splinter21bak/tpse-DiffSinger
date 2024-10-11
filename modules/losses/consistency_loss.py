import torch
import torch.nn as nn
from torch import Tensor


class VelocityConsistencyLoss(nn.Module):
    def __init__(self, loss_type, log_norm=False):
        super().__init__()
        self.loss_type = loss_type
        self.log_norm = log_norm
        if self.log_norm:
            raise ValueError(f'log_norm is not support on consistency model')
        #if self.loss_type == 'l1':
        #    self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_non_padding(v_pred_a, v_pred_b, non_padding=None):
        if non_padding is not None:
            non_padding = non_padding.transpose(1, 2).unsqueeze(1)
            return v_pred_a * non_padding, v_pred_b * non_padding
        else:
            return v_pred_a, v_pred_b

    def _forward(self, v_pred_a, v_pred_b):
        return self.loss(v_pred_a, v_pred_b)

    def forward(self, v_pred_a: Tensor, v_pred_b: Tensor, non_padding: Tensor = None) -> Tensor:
        """
        :param v_pred_a: [B, 1, M, T]
        :param v_pred_b: [B, 1, M, T]
        :param non_padding: [B, T, M]
        """
        v_pred_a, v_pred_b = self._mask_non_padding(v_pred_a, v_pred_b, non_padding)
        return self._forward(v_pred_a, v_pred_b).mean()


class TrajectoryConsistencyLoss(nn.Module):
    def __init__(self, loss_type, log_norm=False, consistency_delta_t=0.1):
        super().__init__()
        self.loss_type = loss_type
        self.log_norm = log_norm
        self.consistency_delta_t =consistency_delta_t
        if self.log_norm:
            raise ValueError(f'log_norm is not support on consistency model')
        #if self.loss_type == 'l1':
        #    self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_non_padding(f_pred_a, f_pred_b, non_padding=None):
        if non_padding is not None:
            non_padding = non_padding.transpose(1, 2).unsqueeze(1)
            return f_pred_a * non_padding, f_pred_b * non_padding
        else:
            return f_pred_a, f_pred_b

    def _forward(self, f_pred_a, f_pred_b):
        loss = self.loss(f_pred_a, f_pred_b) / self.consistency_delta_t ** 2
        return loss

    def forward(self, f_pred_a: Tensor, f_pred_b: Tensor, non_padding: Tensor = None) -> Tensor:
        """
        :param f_pred_a: [B, 1, M, T]
        :param f_pred_b: [B, 1, M, T]
        :param non_padding: [B, T, M]
        """
        f_pred_a, f_pred_b = self._mask_non_padding(f_pred_a, f_pred_b, non_padding)
        return self._forward(f_pred_a, f_pred_b).mean()
