import yaml

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class FocalLoss(torch.nn.Module):
    """Copied from: https://github.com/AdeelH/pytorch-multi-class-focal-loss
    Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = torch.nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        device = x.device
        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.0, device=device)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        # AMP note: callers may mask invalid classes with -1e4. -1e4 is within
        # fp16 range (±65504), so ``log_softmax`` cannot overflow to -inf; the
        # resulting loss stays large-but-finite, never NaN/inf.
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class HierarchicalLoss(torch.nn.Module):
    """Coarse-grained classification loss using cell type taxonomy.

    Loads a YAML mapping fine→coarse cell types, aggregates softmax
    probabilities within each coarse group, and computes NLL loss on
    the coarse labels.

    Notes on usage:
        - Dormant in the canonical recipe (``hierarchical_weight=0``);
          kept for optional auxiliary loss experiments.
        - Expects ``ct2idx`` with 0-indexed values. The identity remap
          (``compact_idx = fine_idx``) suffices — no compact/raw translation
          is needed.
        - Unknown fine names are bucketed into a shared ``"Other"`` coarse
          class.
    """

    def __init__(self, config_path, ct2idx, weight=0.5):
        super().__init__()
        self.weight = weight

        with open(config_path) as f:
            fine_to_coarse = yaml.safe_load(f)

        coarse_names = sorted(set(fine_to_coarse.values()))
        coarse2idx = {name: i for i, name in enumerate(coarse_names)}
        self.n_coarse = len(coarse_names)

        # ct2idx values are 0-indexed.
        n_fine = len(ct2idx)
        fine_to_coarse_idx = torch.zeros(n_fine, dtype=torch.long)
        for fine_name, fine_idx in ct2idx.items():
            compact_idx = fine_idx
            if 0 <= compact_idx < n_fine:
                coarse_name = fine_to_coarse.get(fine_name, "Other")
                fine_to_coarse_idx[compact_idx] = coarse2idx[coarse_name]

        self.register_buffer("fine_to_coarse_idx", fine_to_coarse_idx)
        self.coarse_loss_fn = torch.nn.NLLLoss()

    def forward(self, ct_logits, ct_targets):
        # Compute the coarse-probability aggregation and log in FP32
        # regardless of AMP. Under AMP fp16, scatter-summed probabilities
        # for rare coarse classes can underflow below the fp16 subnormal
        # floor (~6e-8); ``clamp(min=1e-8)`` is below that floor so
        # ``torch.log`` can emit -inf and poison the loss. Casting to fp32
        # before clamp+log keeps the path numerically safe.
        with torch.amp.autocast("cuda", enabled=False):
            fine_probs = F.softmax(ct_logits.float(), dim=-1)
            coarse_probs = torch.zeros(
                fine_probs.shape[0],
                self.n_coarse,
                device=fine_probs.device,
                dtype=fine_probs.dtype,
            )
            coarse_probs.scatter_add_(
                1,
                self.fine_to_coarse_idx.unsqueeze(0).expand_as(fine_probs),
                fine_probs,
            )
            coarse_targets = self.fine_to_coarse_idx[ct_targets]
            log_coarse_probs = torch.log(coarse_probs.clamp(min=1e-7))
            return self.weight * self.coarse_loss_fn(log_coarse_probs, coarse_targets)
