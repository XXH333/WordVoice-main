import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def tpr_loss(disc_real_outputs, disc_generated_outputs, tau):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        m_DG = torch.median((dr - dg))
        L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss


def mel_loss(real_speech, generated_speech, mel_transforms):
    loss = 0
    for transform in mel_transforms:
        mel_r = transform(real_speech)
        mel_g = transform(generated_speech)
        loss += F.l1_loss(mel_g, mel_r)
    return loss


class DynamicStyleLoss(nn.Module):
    def __init__(
        self, 
        max_bnd_class, 
        max_tone_class, 
        max_f0_class, 
        max_energy_class, 
        max_dur_class,
        ignore_id=-2
    ):
        super().__init__()
        self.ignore_id = ignore_id
        
        # 保存各类的数量，用于 view 展平时使用
        self.max_bnd_class = max_bnd_class
        self.max_tone_class = max_tone_class
        self.max_f0_class = max_f0_class
        self.max_energy_class = max_energy_class
        self.max_dur_class = max_dur_class

        # 1. 初始化 Boundary 权重
        bnd_weights = torch.ones(max_bnd_class, dtype=torch.float32)
        if max_bnd_class > 4:
            bnd_weights[1] = 3.
            bnd_weights[2] = 5.
            bnd_weights[3] = 5.
            bnd_weights[4] = 5.
        self.register_buffer('bnd_weights', bnd_weights)

        # 2. 初始化 Tone 权重
        tone_weights = torch.ones(max_tone_class, dtype=torch.float32)
        if max_tone_class > 4:
            tone_weights[1] = 3.
            tone_weights[2] = 2.
            tone_weights[3] = 3.
            tone_weights[4] = 3.
            tone_weights[5] = 5.
            tone_weights[6] = 3.
        self.register_buffer('tone_weights', tone_weights)

        # 3. 初始化 F0 权重
        f0_weights = torch.ones(max_f0_class, dtype=torch.float32)
        # 0~4 指数递减: 20, 10, ..., 1
        start, end, steps = 5, 1.2, 4
        ratio = (end / start) ** (1 / (steps - 1))
        f0_weights[0:steps] = torch.tensor([start * (ratio ** i) for i in range(steps)], dtype=torch.float32)
        # 15~19 指数递增: 1, 2, 4, ..., 20
        start, end, steps = 1.2, 5, 5
        ratio = (end / start) ** (1 / (steps - 1))
        f0_weights[15:15+steps] = torch.tensor([start * (ratio ** i) for i in range(steps)], dtype=torch.float32)
        self.register_buffer('f0_weights', f0_weights)

        # 4. 初始化 Energy 权重
        eng_weights = torch.ones(max_energy_class, dtype=torch.float32)
        # 线性递减
        eng_weights[0] = 6
        eng_weights[1] = 6
        eng_weights[2] = 6
        eng_weights[3:8] = torch.linspace(4, 1.2, steps=5, dtype=torch.float32)
        # 16~19 指数递增
        start, end, steps = 1.2, 10, 4
        ratio = (end / start) ** (1 / (steps - 1))
        eng_weights[16:16+steps] = torch.tensor([start * (ratio ** i) for i in range(steps)], dtype=torch.float32)
        self.register_buffer('eng_weights', eng_weights)

        # 5. 初始化 duration 权重
        dur_weights = torch.ones(max_dur_class, dtype=torch.float32)
        dur_weights[0] = 20.
        dur_weights[1] = 3.
        dur_weights[10:max_dur_class-1] = torch.linspace(1.2, 5, steps=max_dur_class-11, dtype=torch.float32)
        self.register_buffer('dur_weights', dur_weights)
    
    def boundary_loss(self, bnd_pred, bnd_target):
        pred_flat = bnd_pred.view(-1, self.max_bnd_class)
        target_flat = bnd_target.view(-1)
        return F.cross_entropy(pred_flat, target_flat, weight=self.bnd_weights, ignore_index=self.ignore_id)

    def tone_loss(self, tone_pred, tone_target):
        pred_flat = tone_pred.view(-1, self.max_tone_class)
        target_flat = tone_target.view(-1)
        return F.cross_entropy(pred_flat, target_flat, weight=self.tone_weights, ignore_index=self.ignore_id)

    def f0_loss(self, f0_pred, f0_target):
        pred_flat = f0_pred.view(-1, self.max_f0_class)
        target_flat = f0_target.view(-1)
        return F.cross_entropy(pred_flat, target_flat, weight=self.f0_weights, ignore_index=self.ignore_id)

    def energy_loss(self, energy_pred, energy_target):
        pred_flat = energy_pred.view(-1, self.max_energy_class)
        target_flat = energy_target.view(-1)
        return F.cross_entropy(pred_flat, target_flat, weight=self.eng_weights, ignore_index=self.ignore_id)
    
    def duration_loss(self, dur_pred, dur_target):
        pred_flat = dur_pred.view(-1, self.max_dur_class)
        target_flat = dur_target.view(-1)
        return F.cross_entropy(pred_flat, target_flat, weight=self.dur_weights, ignore_index=self.ignore_id)


class DPOLoss(torch.nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards
