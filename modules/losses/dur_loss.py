import torch
import torch.nn as nn
from torch import Tensor


class DurationLoss(nn.Module):
    """
    Loss module as combination of phone duration loss, word duration loss and sentence duration loss.
    """

    def __init__(self, offset, loss_type,
                 lambda_pdur=0.6, lambda_wdur=0.3, lambda_sdur=0.1):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'huber':
            self.loss = nn.HuberLoss()
        else:
            raise NotImplementedError()
        self.offset = offset

        self.lambda_pdur = lambda_pdur
        self.lambda_wdur = lambda_wdur
        self.lambda_sdur = lambda_sdur
        
        # scale太高会nan
        # 这个loss很慢但是很好学，所以只要练好以后稍微跑2-4k steps就好了
        self.use_rules_loss = True
        if self.use_rules_loss:
            self.rules_loss = RulesLoss(offset=self.offset, loss_type=self.loss_type, scale=0.3)
        else:
            self.rules_loss = None

    def linear2log(self, any_dur):
        return torch.log(any_dur + self.offset)

    def forward(self, dur_pred: Tensor, dur_gt: Tensor, ph2word: Tensor, txt_tokens: Tensor) -> Tensor:
        dur_gt = dur_gt.to(dtype=dur_pred.dtype)

        # rules_loss
        if self.use_rules_loss:
            rules_loss = self.rules_loss(dur_pred, txt_tokens)
        else:
            rules_loss = torch.tensor(0.0, device=dur_pred.device)

        # pdur_loss
        pdur_loss = self.lambda_pdur * self.loss(self.linear2log(dur_pred), self.linear2log(dur_gt))

        dur_pred = dur_pred.clamp(min=0.)  # clip to avoid NaN loss

        # wdur loss
        shape = dur_pred.shape[0], ph2word.max() + 1
        wdur_pred = dur_pred.new_zeros(*shape).scatter_add(
            1, ph2word, dur_pred
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        wdur_gt = dur_gt.new_zeros(*shape).scatter_add(
            1, ph2word, dur_gt
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        wdur_loss = self.lambda_wdur * self.loss(self.linear2log(wdur_pred), self.linear2log(wdur_gt))

        # sdur loss
        sdur_pred = dur_pred.sum(dim=1)
        sdur_gt = dur_gt.sum(dim=1)
        sdur_loss = self.lambda_sdur * self.loss(self.linear2log(sdur_pred), self.linear2log(sdur_gt))

        # combine
        dur_loss = pdur_loss + wdur_loss + sdur_loss + rules_loss

        return dur_loss


class RulesLoss(nn.Module):
    def __init__(self, offset, loss_type, scale=0.6):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'huber':
            self.loss = nn.HuberLoss()
        else:
            raise NotImplementedError()
        self.offset = offset
        
        self.scale = scale
        
        # 实际上只有中文存在复合元音产生的口胡问题，所以只处理中文的情况
        # token的值需要参考导出的exp_name.phonemes.json
        self.initials_and_voice_duration = {"94": 2, "122": 3, "100": 2, "92": 2, "43": 5, "27": 5}
        self.ratio_change = ["44", "28", "29", "27", "121", "43"]

    def linear2log(self, any_dur):
        return torch.log(any_dur + self.offset)

    def get_dur_rules(self, dur_pred, txt_tokens):
        dur_rules = torch.zeros_like(dur_pred)

        for b in range(dur_pred.size(0)):
            for i in range(dur_pred.size(1)):
                dur_rules[b, i] = dur_pred[b, i]
                
                token = str(txt_tokens[b, i].item())
                
                if token in self.initials_and_voice_duration:
                    expected_dur = self.initials_and_voice_duration[token]
                    frame = dur_pred[b, i]
                    if frame - expected_dur > 0:
                        gap = frame - expected_dur
                        dur_rules[b, i] = dur_pred[b, i] - gap
                        if i > 0:
                            dur_rules[b, i - 1] = dur_rules[b, i - 1] + gap
                
                if token in self.ratio_change and i + 1 < dur_pred.size(1):
                    frame_before = dur_pred[b, i]
                    frame_post = dur_pred[b, i + 1]
                    if 3 * frame_before > frame_post:
                        gap = frame_before - (frame_post / 3)
                        dur_rules[b, i] = dur_pred[b, i] - gap
                        if i > 0:
                            dur_rules[b, i - 1] = dur_rules[b, i - 1] + gap
                        dur_rules[b, i + 1] = dur_pred[b, i + 1] + gap

        return dur_rules

    def forward(self, dur_pred, txt_tokens):
        dur_rules = self.get_dur_rules(dur_pred, txt_tokens)
        rules_loss = self.scale * self.loss(self.linear2log(dur_pred), self.linear2log(dur_rules))
        
        return rules_loss

