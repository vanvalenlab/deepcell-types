import torch
import torch.nn.functional as F


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


class BCELabelSmoothLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(BCELabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        smoothed_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = F.binary_cross_entropy_with_logits(input, smoothed_target)
        return loss



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