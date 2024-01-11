import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_iterations: int,
        decay_iterations: int,
        max_iterations: int,
        eta_min: float = 0.0,
        last_iteration: int = -1,
    ) -> None:
        self.warmup_iterations: int = warmup_iterations
        self.decay_iterations: int = decay_iterations
        self.max_iterations: int = max_iterations
        self.eta_min: float = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch=last_iteration)

    def get_lr(self):
        if self.last_epoch < self.warmup_iterations:
            # Linear warmup
            warmup_ratio: float = self.last_epoch / self.warmup_iterations
            return [
                max(self.eta_min + (base_lr - self.eta_min) * warmup_ratio, self.eta_min)
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch < self.decay_iterations:
            # Cosine annealing
            progress: float = (self.last_epoch - self.warmup_iterations) / (
                self.decay_iterations - self.warmup_iterations
            )
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress) * 3.14159))
            return [
                max(self.eta_min + (base_lr - self.eta_min) * cosine_decay.item(), self.eta_min)
                for base_lr in self.base_lrs
            ]
        else:
            # Constant learning rate after decay_iterations
            return [self.eta_min for _ in self.base_lrs]
