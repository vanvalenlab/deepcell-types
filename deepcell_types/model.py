import numpy as np
import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class CellTypeClassificationHead(nn.Module):
    def __init__(self, n_filters, n_celltypes, dropout_rate=0.1):
        super(CellTypeClassificationHead, self).__init__()
        self.dense1 = nn.Linear(n_filters, n_filters)
        self.dense2 = nn.Linear(n_filters, n_filters // 2)
        self.dense3 = nn.Linear( n_filters // 2, n_celltypes)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = x
        out = self.dense1(out)
        out = self.silu(out)
        out = self.dropout(out)
        out = self.dense2(out)
        out = self.silu(out)
        out = self.dropout(out)
        out = self.dense3(out)
        return out


class DomainClassificationHead(nn.Module):
    """For domain adaptation, reverse the gradient."""
    def __init__(self, n_filters, n_domains, dropout_rate=0.1):
        super(DomainClassificationHead, self).__init__()
        self.dense1 = nn.Linear(n_filters, n_filters)
        self.dense2 = nn.Linear(n_filters, n_filters // 2)
        self.dense3 = nn.Linear(n_filters // 2, n_domains)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = x
        out = grad_reverse(out)
        out = self.dense1(out)
        out = self.silu(out)
        out = self.dropout(out)
        out = self.dense2(out)
        out = self.silu(out)
        out = self.dropout(out)
        out = self.dense3(out)
        return out


class ConvBlock(nn.Module):
    """ Simple Convolutional block for feature extraction """
    def __init__(self, n_filters):
        super(ConvBlock, self).__init__()
        self.n_filters = n_filters

        self.layers = nn.Sequential(
            nn.Conv2d(3, n_filters//16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filters//16),
            nn.SiLU(),
            nn.Conv2d(n_filters//16, n_filters//16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters//16),
            nn.SiLU(),
            nn.Conv2d(n_filters//16, n_filters//8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filters//8),
            nn.SiLU(),
            nn.Conv2d(n_filters//8, n_filters//8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters//8),
            nn.SiLU(),
            nn.Conv2d(n_filters//8, n_filters//4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filters//4),
            nn.SiLU(),
            nn.Conv2d(n_filters//4, n_filters//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters//4),
            nn.SiLU(),
            nn.Conv2d(n_filters//4, n_filters//2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filters//2),
            nn.SiLU(),
            nn.Conv2d(n_filters//2, n_filters//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters//2),
            nn.SiLU(),
            nn.Conv2d(n_filters//2, n_filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.SiLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.SiLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.SiLU(),
        )

    def forward(self, x):
        # merget first two dimensions (B, C, 3, H, W) -> (B * C, 3, H, W)
        out = x.view(-1, 3, x.shape[-2], x.shape[-1])
        out = self.layers(out)

        # reshape back to original shape
        assert out.shape[-2] == out.shape[-1] == 1  # spatial dimensions are 1
        out = out.view(x.shape[0], x.shape[1], self.n_filters)
        return out


class MarkerNameEmbeddingLayer(nn.Module):
    """Load pre-trained embeddings for marker names, then apply a linear layer."""
    def __init__(self, n_filters, marker_embeddings):
        super(MarkerNameEmbeddingLayer, self).__init__()

        embeddings = torch.cat(
            [
                torch.zeros(1, marker_embeddings.shape[1]),  # padding
                torch.as_tensor(marker_embeddings),
            ],
            dim=0,
        )

        self.embed_layer = nn.Embedding.from_pretrained(
            embeddings, freeze=True, padding_idx=0
        )
        self.dense = nn.Linear(embeddings.shape[1], n_filters)

    def forward(self, x):
        out = x + 1  # shift by 1 to account for padding
        out = self.embed_layer(out)
        out = self.dense(out)
        return out



class CellTypeDataEncoder(nn.Module):
    """ Encode cell type data, including marker names and images. """
    def __init__(self, n_filters, n_heads, n_celltypes, n_domains, marker_embeddings, img_feature_extractor):
        super(CellTypeDataEncoder, self).__init__()
        self.n_heads = n_heads
        self.n_celltypes = n_celltypes
        self.n_domains = n_domains

        # Define marker name embedding layer
        self.marker_embedder = MarkerNameEmbeddingLayer(n_filters, marker_embeddings)

        # Define CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_filters))

        # Define blocks
        self.img_feature_extractor = ConvBlock(n_filters)
        
        self.transformer_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_filters, nhead=n_heads, dim_feedforward=n_filters*2, batch_first=True
            ),
            num_layers=5,
        )
        self.classification_head = CellTypeClassificationHead(n_filters, n_celltypes)
        self.domain_classification_head = DomainClassificationHead(n_filters, n_domains)
        self.cls_single_attention = nn.MultiheadAttention(n_filters, num_heads=1, dropout=0.0, batch_first=True)

        self.marker_positivity_head = nn.Linear(n_filters, 1)

    def forward(self, inputs_app, inputs_ch_names, inputs_ch_padding_masks):
        """
        inputs_app: (B, C, 3, H, W)
        inputs_ch_padding_mask: (B, C), True=ignore
        """
        aug_inputs_ch_padding_masks = nn.functional.pad(
            inputs_ch_padding_masks.long(), (1, 0), mode="reflect"
        ).bool()  # (B, C+1) - add padding for CLS token

        # Apply convolutions
        x = self.img_feature_extractor(inputs_app)  # (B, C, n_filters)

        # Create marker name embeddings
        marker_embeddings = self.marker_embedder(inputs_ch_names)
        
        if self.training:
            # Add noise to marker name embeddings
            marker_embeddings = marker_embeddings + torch.randn_like(marker_embeddings) * 0.005
            # Normalize marker embeddingste
            marker_embeddings = marker_embeddings / marker_embeddings.norm(dim=-1, keepdim=True)

        x = x + marker_embeddings

        # Apply transformer (w/o CLS token)
        x = self.transformer_blocks(x, src_key_padding_mask=inputs_ch_padding_masks)

        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, C+1, n_filters)

        # Apply Single attention layer
        x, attention = self.cls_single_attention(x, x, x, key_padding_mask=aug_inputs_ch_padding_masks, need_weights=True, average_attn_weights=False)
        
        # Take the CLS token embedding out
        cls_token_embedding = x[:, 0, :]  # (B, n_filters)

        # Apply classification heads
        celltype_output = self.classification_head(cls_token_embedding)
        domain_output = self.domain_classification_head(cls_token_embedding)

        return celltype_output, domain_output, cls_token_embedding, attention[:, 0, 0, 1:]

class CellTypeCLIPModel(nn.Module):
    """ Apply contrastive learning to data against cell type names. """
    def __init__(self, n_filters, embedding_dim, ct_embeddings, marker_embeddings, n_heads,n_celltypes, n_domains, img_feature_extractor="conv"):
        super(CellTypeCLIPModel, self).__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.ct_embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(ct_embeddings), freeze=True
        )

        self.image_encoder = CellTypeDataEncoder(
            n_filters=n_filters, 
            n_heads=n_heads, 
            n_celltypes=n_celltypes, 
            n_domains=n_domains, 
            marker_embeddings=marker_embeddings,
            img_feature_extractor=img_feature_extractor,
        )

        self.image_adaptor = nn.Sequential(
            nn.Linear(n_filters, n_filters),
        )
        self.text_adaptor = nn.Linear(embedding_dim, n_filters)


    def forward(self, sample, ch_idx, mask):

        # Encode image
        _, _, cls_token_embedding, marker_pos_attn = self.image_encoder(
            sample, ch_idx, mask
        )
        image_embedding = cls_token_embedding
        image_embedding = self.image_adaptor(image_embedding)

        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()

        # extract probabilities for each image
        raw_text_embedding_all_classes = self.ct_embedding.weight # shape = [n_celltypes, embedding_dim]
        text_embedding_all_classes = self.text_adaptor(raw_text_embedding_all_classes)
        text_embedding_all_classes = text_embedding_all_classes / text_embedding_all_classes.norm(dim=-1, keepdim=True)
        logits_per_image_all_classes = logit_scale * image_embedding @ text_embedding_all_classes.t()
        probs = torch.softmax(logits_per_image_all_classes, dim=-1) # shape = [global_batch_size, n_celltypes]

        # normalize marker_pos_attn by max value
        marker_pos_attn = marker_pos_attn / torch.max(marker_pos_attn, dim=-1, keepdim=True)[0]

        return None, None, None, marker_pos_attn, probs, image_embedding
