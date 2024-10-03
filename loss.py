import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # Normalize the features to unit vectors
        features = F.normalize(features, p=2, dim=1)
        
        # Compute the similarity matrix
        similarity_matrix = torch.matmul(features, features.T)
        
        # Apply the temperature scaling
        similarity_matrix /= self.temperature
        
        # Get the labels matrix
        labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

        # Exclude the main diagonal from the loss
        labels_matrix = labels_matrix.fill_diagonal_(0)

        # Compute the numerator
        numerator = torch.exp(similarity_matrix * labels_matrix).sum(axis=1)
        
        # Compute the denominator
        mask_non_self = (1 - torch.eye(labels_matrix.shape[0])).to(labels.device)
        demoninator = torch.exp(similarity_matrix * mask_non_self).sum(axis=1)

        # Compute the positive pair counts
        P = labels_matrix.sum(axis=1) # Number of positive pairs for each sample
        mask_non_zero = P > 0

        # Compute the loss
        loss = -torch.log(numerator / demoninator)[mask_non_zero] * 1/P[mask_non_zero]

        return loss.mean()


class PULoss(torch.nn.Module):
    def __init__(self, alpha=0.6):
        super(PULoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        # y_true contains 1 for positive samples, and 0 for unlabeled samples
        # y_pred contains the predicted probabilities

        pos_mask = (y_true == 1).float()
        unlabeled_mask = (y_true == 0).float()

        pos_loss = F.binary_cross_entropy(y_pred, pos_mask, reduction="none")
        unlabeled_loss = F.binary_cross_entropy(
            y_pred, unlabeled_mask, reduction="none"
        )

        pu_loss = self.alpha * torch.mean(pos_loss) + (1 - self.alpha) * torch.mean(
            unlabeled_loss
        )
        return pu_loss


class BCELabelSmoothLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(BCELabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        smoothed_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = F.binary_cross_entropy_with_logits(input, smoothed_target)
        return loss



class FocalLoss(torch.nn.Module):
    """ Copied from: https://github.com/AdeelH/pytorch-multi-class-focal-loss
    Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
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
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = torch.nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()

    def forward(self, logits_per_image, logits_per_text):
        target = torch.arange(logits_per_image.shape[0]).to(logits_per_image.device)
        loss_image = F.cross_entropy(logits_per_image, target)
        loss_text = F.cross_entropy(logits_per_text, target)
        return (loss_image + loss_text) / 2.0
        


class FocalCLIPLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=0):
        super(FocalCLIPLoss, self).__init__()
        self.gamma = gamma
        self.nll_loss = torch.nn.NLLLoss(
            weight=alpha, reduction='none')

    def forward(self, logits_per_image, logits_per_text):
        target = torch.arange(logits_per_image.shape[0]).to(logits_per_image.device)
        
        def focal_loss_on_one_logit(logits):
            log_p = F.log_softmax(logits, dim=-1)
            ce = self.nll_loss(log_p, target)

            # get true class column from each row
            all_rows = torch.arange(len(logits))
            log_pt = log_p[all_rows, target]

            # compute focal term: (1 - pt)^gamma
            pt = log_pt.exp()
            focal_term = (1 - pt)**self.gamma

            # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
            loss = focal_term * ce

            loss = loss.mean()
            return loss

        loss_image = focal_loss_on_one_logit(logits_per_image)
        loss_text = focal_loss_on_one_logit(logits_per_text)

        return (loss_image + loss_text) / 2.0