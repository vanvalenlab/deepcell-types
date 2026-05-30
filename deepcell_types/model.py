import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Shared utilities
# =============================================================================


class GradReverse(torch.autograd.Function):
    """Gradient-reversal for DANN-style domain adversarial training.

    Forward is identity; backward multiplies the incoming gradient by -1.
    The DANN reversal factor ``λ`` is hardcoded to 1 (no schedule). The
    canonical training recipe enables this branch via
    ``--domain_weight 0.1`` (the current ``train.py`` default); setting
    ``--domain_weight 0`` disables it entirely. The fixed λ has been
    sufficient in practice at this weighting; if the head is ever weighted
    more aggressively, consider adding a ramp schedule (Ganin & Lempitsky
    2015, eq. 12).
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


# =============================================================================
# Architecture: CellTypeAnnotator
# =============================================================================


class SpatialEncoder(nn.Module):
    """3-layer CNN processing shared spatial context (self_mask, neighbor_mask, distance_transform).

    Input: (B, 3, H, W)
    Output: (B, out_dim * pool_size * pool_size)  -- default (B, 64)

    Uses BatchNorm2d. InstanceNorm2d was previously tried for OOD robustness but
    InstanceNorm + AdaptiveAvgPool2d(1) zeros per-sample spatial means, losing cell
    size/density information after the global pool. BatchNorm preserves these means.
    """

    def __init__(self, out_dim=64, pool_size=1):
        super().__init__()
        self.out_features = out_dim * pool_size * pool_size
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(pool_size),
        )

    def forward(self, x):
        """x: (B, 3, H, W) -> (B, out_features)"""
        out = self.layers(x)
        return out.view(x.shape[0], -1)


class ResBlock(nn.Module):
    """Residual block for per-channel processing.

    Uses InstanceNorm2d instead of BatchNorm2d because the ResNet processes
    (B*C_max) samples at once, ~73% of which are all-zero padding channels.
    BatchNorm running stats get corrupted by the zero-padding inputs, causing
    wrong normalization during eval. InstanceNorm normalizes each sample
    independently, so padding channels don't contaminate real channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(channels, affine=True)
        self.silu = nn.SiLU()

    def forward(self, x):
        residual = x
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.silu(out + residual)
        return out


class PerChannelResNet(nn.Module):
    """3 ResBlocks processing each marker channel independently.

    Input: (B, C_max, 1, H, W) - raw * self_mask
    Output: (B, C_max, out_dim)  -- default (B, C_max, 128)
    """

    def __init__(self, out_dim=128, base_channels=32):
        super().__init__()
        self.out_dim = out_dim
        self.base_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.SiLU(),
        )
        self.block1 = ResBlock(base_channels)
        self.block2 = ResBlock(base_channels)
        self.block3 = ResBlock(base_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(base_channels, out_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C_max, 1, H, W) - raw * self_mask per channel

        Returns:
            (B, C_max, out_dim)
        """
        B, C_max, _, H, W = x.shape
        out = x.reshape(B * C_max, 1, H, W)
        out = self.stem(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.pool(out)  # (B*C_max, base_channels, 1, 1)
        out = out.view(B, C_max, self.base_channels)
        out = self.proj(out)  # (B, C_max, out_dim)
        return out


class MarkerEmbeddingLayer(nn.Module):
    """Marker name embeddings with ALWAYS-normalized output.

    Uses pre-computed SVD-reduced embeddings: canonical checkpoint is
    ``embeddings/svd_512.npz`` (278 markers x 328-d, reduced from 3072-d
    OpenAI text-embedding-3-large vectors). Always normalizes after projection
    (no train/eval mismatch).
    """

    def __init__(self, d_model, marker_embeddings):
        super().__init__()

        # Add padding embedding at index 0
        embed_dim = marker_embeddings.shape[1]
        # Use .tolist() to avoid torch/numpy C-API version incompatibilities
        embeddings = torch.cat(
            [
                torch.zeros(1, embed_dim),  # padding
                torch.tensor(
                    np.asarray(marker_embeddings).tolist(), dtype=torch.float32
                ),
            ],
            dim=0,
        )

        self.embed_layer = nn.Embedding.from_pretrained(
            embeddings, freeze=True, padding_idx=0
        )
        self.proj = nn.Linear(embed_dim, d_model)

    def forward(self, ch_idx):
        """ch_idx: (B, C_max) -> (B, C_max, d_model), always normalized.

        Padding positions (ch_idx == -1) get exactly zero output so that
        ``proj.bias`` contamination cannot reach the transformer for masked
        tokens.
        """
        out = ch_idx + 1  # shift for padding
        raw_emb = self.embed_layer(out)
        out = self.proj(raw_emb)
        # Always normalize (no train/eval mismatch)
        out = F.normalize(out, p=2, dim=-1)
        # Zero padding positions: F.normalize of a zero vector is zero, but
        # the post-proj output for padding is proj.bias (non-zero after
        # training), which would otherwise carry a unit-norm direction
        # into the transformer for masked channels.
        out = out * (ch_idx != -1).unsqueeze(-1).to(out.dtype)
        return out


class ChannelWiseTransformerEncoderLayer(nn.Module):
    """Channel-wise transformer encoder layer (pre-norm).

    Operates over marker/channel tokens (each channel is a token, plus the
    prepended CLS). Internally uses the pre-norm (Xiong-2020) convention:
    LayerNorm is applied to each sublayer's input, and dropout is applied to
    the sublayer output before the residual add (both attention and FF). The
    FF branch already terminates in ``nn.Dropout`` inside ``self.ff``; the
    attention branch uses a dedicated ``self.attn_dropout`` module (the
    ``dropout`` arg of ``nn.MultiheadAttention`` only drops attention weights,
    not the sublayer output). Pre-norm keeps the residual stream a clean
    identity highway, which trains more stably than post-norm.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, src_key_padding_mask=None, return_attn_weights=False):
        # Pre-norm attention
        normed = self.norm1(x)
        if return_attn_weights:
            attn_out, attn_weights = self.attn(
                normed,
                normed,
                normed,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
        else:
            attn_out, _ = self.attn(
                normed, normed, normed, key_padding_mask=src_key_padding_mask
            )
            attn_weights = None
        x = x + self.attn_dropout(attn_out)
        # Pre-norm feedforward
        x = x + self.ff(self.norm2(x))
        if return_attn_weights:
            return x, attn_weights
        return x


class CellTypeAnnotator(nn.Module):
    """Cell type annotator with marker-aware transformer.

    Architecture:
        1. SpatialEncoder: (B, 3, H, W) -> (B, 64) - process masks ONCE
        2. PerChannelResNet: (B, C_max, 1, H, W) -> (B, C_max, 128) - per-marker features
        3. Fusion: cat(channel_feat, spatial_feat) -> Linear -> (B, C_max, d_model)
        4. MarkerEmbeddings: always-normalized, added to fused features
        5. Pre-norm Transformer: CLS prepended, 4 layers, d=256, ff=1024, 8 heads
        6. Task heads: cell type (FocalLoss), marker positivity (independent sigmoid), domain (GradReverse)
    """

    def __init__(
        self,
        d_model=256,
        n_heads=8,
        n_layers=4,
        n_celltypes=51,
        n_domains=8,
        marker_embeddings=None,
        dropout=0.1,
        tumor_head=False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_celltypes = n_celltypes
        self.n_domains = n_domains

        # 1. Spatial encoder
        spatial_pool_size = kwargs.get("spatial_pool_size", 1)
        self.spatial_encoder = SpatialEncoder(out_dim=64, pool_size=spatial_pool_size)
        spatial_dim = self.spatial_encoder.out_features

        # 2. Per-channel feature extractor
        channel_dim = 128
        # Canonical paper recipe is base_channels=48 (matches CLI default in
        # scripts/train.py and scripts/predict.py).
        resnet_base_channels = kwargs.get("resnet_base_channels", 48)
        self.channel_encoder = PerChannelResNet(
            out_dim=channel_dim, base_channels=resnet_base_channels
        )

        # 3. Fusion layer: concat channel features (128) + spatial features (64) -> d_model
        self.fusion = nn.Linear(channel_dim + spatial_dim, d_model)

        # 4. Marker name embeddings (LoRA removed — redundant with trainable proj)
        if marker_embeddings is not None:
            self.marker_embedder = MarkerEmbeddingLayer(d_model, marker_embeddings)
        else:
            self.marker_embedder = None

        # 5. CLS token (prepended BEFORE transformer)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Pre-norm transformer
        self.transformer_layers = nn.ModuleList(
            [
                ChannelWiseTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # 6. Task heads
        # Cell type: MLP classifier
        self.ct_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_celltypes),
        )

        # Marker positivity: default head (replaced if MarkerConditionedMPHead is used)
        self.marker_pos_head = nn.Linear(d_model, 1)

        # Domain: GradReverse + LayerNorm (not BatchNorm)
        self.domain_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_domains),
        )

        # Tumor: optional binary prediction head
        self._has_tumor_head = tumor_head
        if tumor_head:
            self.tumor_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1),
            )

        # Mean-intensity-per-channel side input (zero-init → identity warm-start).
        # Canonical paper recipe is "cls_residual" (matches CLI default in
        # scripts/train.py). Older "none" was the pre-MeanInt-CLS default.
        self.mean_intensity_mode = kwargs.get("mean_intensity_mode", "cls_residual")
        # v0.1.0 checkpoints were trained with a scatter that aliased padding
        # writes to column 0 of intensity_vec, so the model never saw the real
        # mean intensity of whichever marker sits at index 0 in marker2idx.
        # The fixed code (sink column) now routes those writes elsewhere. To
        # keep inference parity with the v0.1.0 canonical checkpoint, we
        # explicitly zero column 0 at forward time. Set to False when
        # retraining from scratch (v0.2.0+) to recover the marker-0 signal.
        self.compat_marker0_zero = bool(kwargs.get("compat_marker0_zero", True))
        if self.mean_intensity_mode != "none":
            n_markers = (
                marker_embeddings.shape[0] if marker_embeddings is not None else 278
            )
            self._n_markers = n_markers
            if self.mean_intensity_mode in ("cls_residual", "both"):
                self.intensity_cls_branch = nn.Sequential(
                    nn.Linear(n_markers, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                )
                nn.init.zeros_(self.intensity_cls_branch[-1].weight)
                nn.init.zeros_(self.intensity_cls_branch[-1].bias)
            if self.mean_intensity_mode in ("per_channel", "both"):
                self.intensity_per_channel_proj = nn.Linear(1, d_model)
                nn.init.zeros_(self.intensity_per_channel_proj.weight)
                nn.init.zeros_(self.intensity_per_channel_proj.bias)

    def forward(
        self,
        sample,
        spatial_context,
        ch_idx,
        padding_mask,
        ct_exclude=None,
        return_attn_weights=False,
        domain_idx=None,
    ):
        """
        Args:
            sample: (B, C_max, 1, H, W) - raw intensity * self_mask per channel
            spatial_context: (B, 3, H, W) - [self_mask, neighbor_mask, distance_transform]
            ch_idx: (B, C_max) - channel indices for marker embeddings
            padding_mask: (B, C_max) - True = padding channel
            ct_exclude: Optional list of lists of excluded ct indices per sample
            return_attn_weights: If True, return CLS→channel attention weights
            domain_idx: (B,) long tensor - unused at the per-channel encoder level
                (kept in signature because training scripts pass it); domain info
                only enters the model via the DANN head (grad_reverse on CLS).

        Returns:
            ct_logits: (B, n_celltypes) - cell type logits
            domain_logits: (B, n_domains) - domain logits
            marker_pos_logits: (B, C_max) - marker positivity logits (pre-sigmoid)
            cls_embedding: (B, d_model) - CLS token embedding
            channel_outputs: (B, C_max, d_model) - per-channel transformer outputs
            (optional) cls_to_channels: (n_layers, B, C_max) - CLS→channel attention
        """
        B, C_max = sample.shape[0], sample.shape[1]

        # 1. Spatial encoding (shared, process masks ONCE)
        spatial_feat = self.spatial_encoder(spatial_context)  # (B, 64)

        # 2. Per-channel encoding
        channel_feat = self.channel_encoder(sample)  # (B, C_max, 128)
        # Zero out padded channel features to prevent them from affecting downstream computation.
        # Use out-of-place masked_fill to avoid in-place writes on tensors in
        # the backward graph under AMP autocast.
        channel_feat = channel_feat.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # 3. Fusion: broadcast spatial features across channels and concatenate.
        # Zero spatial features for padding positions BEFORE concat. Without
        # this, padding tokens enter ``self.fusion`` with [0, spatial_feat] and
        # emerge as ``W_spatial @ spatial_feat + bias`` — non-trivial features
        # for tokens that should be invisible to the transformer.
        # Use out-of-place masked_fill (same AMP-safe pattern as channel_feat
        # above); the expand() view is materialized by masked_fill.
        spatial_expanded = spatial_feat.unsqueeze(1).expand(-1, C_max, -1)
        spatial_expanded = spatial_expanded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        fused = torch.cat([channel_feat, spatial_expanded], dim=-1)  # (B, C_max, 192)
        fused = self.fusion(fused)  # (B, C_max, d_model)
        # The fusion linear adds its bias to even the zeroed padding inputs,
        # so padding tokens emerge as ``fusion.bias``. The attention's
        # ``src_key_padding_mask`` makes this invisible to other tokens, but
        # neutralise the bias here as well so the documented invariant
        # ("padding produces zero output") holds at the tensor level.
        fused = fused.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # 4. Add marker name embeddings
        if self.marker_embedder is not None:
            marker_emb = self.marker_embedder(ch_idx)  # (B, C_max, d_model)
            fused = fused + marker_emb

        # 4b. Mean-intensity-per-channel side input (computed from sample masked by self_mask)
        mean_intensity = None
        if self.mean_intensity_mode != "none":
            # sample is raw * self_mask, so sum / cell_size gives mean over the cell footprint
            self_mask_2d = spatial_context[:, 0]  # (B, H, W)
            cell_size = self_mask_2d.sum(dim=(-1, -2)).clamp(min=1.0)  # (B,)
            mean_intensity = sample.sum(dim=(-1, -2, -3)) / cell_size.unsqueeze(
                -1
            )  # (B, C_max)
            mean_intensity = mean_intensity.masked_fill(padding_mask, 0.0)
            if self.mean_intensity_mode in ("per_channel", "both"):
                per_ch = self.intensity_per_channel_proj(
                    mean_intensity.unsqueeze(-1)
                )  # (B, C_max, d_model)
                per_ch = per_ch.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                fused = fused + per_ch

        # 5. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, fused], dim=1)  # (B, C_max+1, d_model)

        # CLS padding mask (CLS is never padded)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
        aug_padding_mask = torch.cat([cls_pad, padding_mask], dim=1)  # (B, C_max+1)

        # Apply pre-norm transformer layers
        all_attn_weights = [] if return_attn_weights else None
        for layer in self.transformer_layers:
            if return_attn_weights:
                x, attn_w = layer(
                    x, src_key_padding_mask=aug_padding_mask, return_attn_weights=True
                )
                # attn_w: (B, seq_len, seq_len) — extract CLS row, skip CLS column
                all_attn_weights.append(attn_w[:, 0, 1:])  # (B, C_max)
            else:
                x = layer(x, src_key_padding_mask=aug_padding_mask)

        x = self.final_norm(x)

        # 6. Extract outputs
        cls_embedding = x[:, 0, :]  # (B, d_model)
        channel_outputs = x[:, 1:, :]  # (B, C_max, d_model)

        # 6b. Mean-intensity-per-channel CLS residual (scatter per-cell
        # intensities to global marker positions).
        #
        # We allocate one extra "sink" column at index ``self._n_markers`` and
        # redirect every padding position's write there. Without the sink,
        # ``safe_idx[~valid] = 0`` aliases all padding writes to column 0,
        # and ``scatter_``'s last-write-wins semantics then overwrite the
        # real mean intensity of whichever marker sits at index 0 with the
        # 0.0 written from padding positions. The sink is sliced off before
        # the projection so the rest of the model sees the intended
        # ``(B, n_markers)`` shape.
        #
        # NOTE (v0.1.0 checkpoint compat): historical checkpoints were trained
        # with a scatter that *did* alias padding to column 0, so column 0
        # was effectively always 0.0 at training time and the projection's
        # column-0 weights only ever saw zero. Feeding them a real intensity
        # value now would shift inference outputs away from the published
        # paper numbers. We explicitly zero column 0 below so the canonical
        # v0.1.0 checkpoint reproduces its training-time outputs bit-for-bit.
        # The flag is read from the model's stored architecture metadata so
        # a future retrain (v0.2.0) can flip it off with ``compat_marker0_zero=False``.
        if (
            self.mean_intensity_mode in ("cls_residual", "both")
            and mean_intensity is not None
        ):
            intensity_vec = torch.zeros(
                B, self._n_markers + 1, device=sample.device, dtype=mean_intensity.dtype
            )
            valid = ~padding_mask
            safe_idx = ch_idx.clone()
            safe_idx[~valid] = self._n_markers  # sink column for padding writes
            masked_int = mean_intensity.masked_fill(~valid, 0.0)
            intensity_vec.scatter_(1, safe_idx, masked_int)
            intensity_vec = intensity_vec[:, : self._n_markers]  # drop sink
            if getattr(self, "compat_marker0_zero", True):
                intensity_vec = intensity_vec.clone()
                intensity_vec[:, 0] = 0.0
            cls_embedding = cls_embedding + self.intensity_cls_branch(intensity_vec)

        # Cell type classification from CLS
        ct_logits = self.ct_head(cls_embedding)  # (B, n_celltypes)

        # Apply tissue-specific exclusion
        if ct_exclude is not None:
            exclude_mask = torch.zeros_like(ct_logits, dtype=torch.bool)
            for i, excl in enumerate(ct_exclude):
                if excl:
                    exclude_mask[i, list(excl)] = True
            # -1e4 is AMP-safe: the smallest finite fp16 value is about
            # -65504, so -1e4 round-trips exactly in fp16 (no overflow to
            # -inf), while exp(-1e4) ≈ 0 makes softmax probability vanish.
            # If the true class is ever filled with -1e4, log_softmax + NLL
            # yields a large-but-finite loss (~1e4) rather than inf.
            ct_logits = ct_logits.masked_fill(exclude_mask, -1e4)

        # Domain classification from CLS (with gradient reversal)
        domain_logits = self.domain_head(grad_reverse(cls_embedding))  # (B, n_domains)

        # Marker positivity from per-channel outputs
        if isinstance(self.marker_pos_head, MarkerConditionedMPHead):
            marker_pos_logits = self.marker_pos_head(
                channel_outputs, ch_idx
            )  # (B, C_max)
        else:
            marker_pos_logits = self.marker_pos_head(channel_outputs).squeeze(
                -1
            )  # (B, C_max)

        # Tumor binary prediction from CLS
        tumor_logit = (
            self.tumor_head(cls_embedding) if self._has_tumor_head else None
        )  # (B, 1) or None

        if return_attn_weights:
            cls_to_channels = torch.stack(
                all_attn_weights, dim=0
            )  # (n_layers, B, C_max)
            return (
                ct_logits,
                domain_logits,
                marker_pos_logits,
                cls_embedding,
                channel_outputs,
                tumor_logit,
                cls_to_channels,
            )

        return (
            ct_logits,
            domain_logits,
            marker_pos_logits,
            cls_embedding,
            channel_outputs,
            tumor_logit,
        )


class MarkerConditionedMPHead(nn.Module):
    """FiLM-conditioned marker positivity head.

    Uses marker embeddings to modulate the decision boundary per marker,
    allowing different thresholds for different markers (e.g., CD3 vs CD45).

    FiLM conditioning:
        scale = sigmoid(linear(marker_emb))
        shift = linear(marker_emb)
        conditioned = outputs * scale + shift

    Then MLP: Linear(d, d//2) → SiLU → Linear(d//2, 1)
    """

    def __init__(self, d_model, marker_embedding_layer):
        super().__init__()
        self.marker_embedding_layer = marker_embedding_layer  # shared, frozen

        # FiLM modulation from marker embeddings
        self.film_scale = nn.Linear(d_model, d_model)
        # Initialize bias to 4.0 so sigmoid(output) ≈ 0.98 (near-identity at init).
        # Initialize weight to zero so the initial scale is purely bias-driven,
        # independent of the marker embedding magnitude.
        nn.init.zeros_(self.film_scale.weight)
        nn.init.constant_(self.film_scale.bias, 4.0)
        self.film_shift = nn.Linear(d_model, d_model)

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, channel_outputs, ch_idx):
        """
        Args:
            channel_outputs: (B, C_max, d_model) - per-channel transformer outputs
            ch_idx: (B, C_max) - channel indices

        Returns:
            (B, C_max) - marker positivity logits (pre-sigmoid)
        """
        # Get marker embeddings (already normalized by MarkerEmbeddingLayer)
        marker_emb = self.marker_embedding_layer(ch_idx)  # (B, C_max, d_model)

        # FiLM conditioning
        scale = torch.sigmoid(self.film_scale(marker_emb))  # (B, C_max, d_model)
        shift = self.film_shift(marker_emb)  # (B, C_max, d_model)
        conditioned = channel_outputs * scale + shift  # (B, C_max, d_model)

        return self.head(conditioned).squeeze(-1)  # (B, C_max)


def create_model(
    dct_config,
    marker_embeddings,
    d_model=256,
    n_heads=8,
    n_layers=4,
    dropout=0.1,
    use_conditioned_mp_head=True,
    **kwargs,
):
    """Factory function to create CellTypeAnnotator from config.

    Args:
        dct_config: DCTConfig or any config object with ``.NUM_CELLTYPES``,
            ``.NUM_DOMAINS``, and ``.marker2idx`` (training callers pass
            ``TissueNetConfig``; inference passes ``DCTConfig``).
        marker_embeddings: numpy array of marker embeddings
        d_model: transformer hidden dimension
        n_heads: number of attention heads
        n_layers: number of transformer layers
        dropout: dropout rate
        use_conditioned_mp_head: If True, replace linear MP head with
            MarkerConditionedMPHead (FiLM-conditioned)
        **kwargs: passed to CellTypeAnnotator (e.g. spatial_pool_size)

    Returns:
        CellTypeAnnotator instance
    """
    n_celltypes = kwargs.pop("n_celltypes", dct_config.NUM_CELLTYPES)
    n_domains = kwargs.pop("n_domains", dct_config.NUM_DOMAINS)
    model = CellTypeAnnotator(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_celltypes=n_celltypes,
        n_domains=n_domains,
        marker_embeddings=marker_embeddings,
        dropout=dropout,
        **kwargs,
    )

    if use_conditioned_mp_head and model.marker_embedder is not None:
        model.marker_pos_head = MarkerConditionedMPHead(d_model, model.marker_embedder)

    return model


class MaskedMarkerHead(nn.Module):
    """Decoder head for masked marker pre-training.

    Predicts mean expression of masked marker channels from transformer outputs.
    Analogous to masked language modeling but for continuous marker intensity values.

    Input: (B, C_max, d_model) per-channel transformer outputs
    Output: (B, C_max) predicted mean expression per channel
    """

    def __init__(self, d_model):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, channel_outputs):
        """channel_outputs: (B, C_max, d_model) -> (B, C_max)"""
        return self.head(channel_outputs).squeeze(-1)


def mask_marker_channels(sample, padding_mask, mask_ratio=0.3, min_keep=1):
    """Randomly mask valid marker channels for pre-training.

    Zeroes out input data for selected channels while preserving their
    position in the sequence (so the model still gets marker embeddings
    for masked channels and must predict their expression from context).

    Args:
        sample: (B, C_max, 1, H, W) - raw intensity per channel
        padding_mask: (B, C_max) - True = padding channel
        mask_ratio: fraction of valid channels to mask
        min_keep: minimum number of unmasked valid channels

    Returns:
        masked_sample: (B, C_max, 1, H, W) - sample with masked channels zeroed
        masked_indices: (B, C_max) bool tensor - True = this channel was masked
        mean_expression: (B, C_max) - mean expression per channel (targets)
    """
    B, C_max = padding_mask.shape

    # Compute mean expression BEFORE masking (targets).
    # sample = raw_intensity * self_mask, so pixels outside the cell are zero.
    # Padding channels are filled with -1.0 by FullImageDataset; ignore them
    # here or every padded pixel would be mistaken for cell area.
    valid_sample = sample.masked_fill(padding_mask.view(B, C_max, 1, 1, 1), 0.0)
    # Averaging over the full patch biases the target low by cell_area/patch_area
    # (~1/1.7x–1/4x); instead, divide the sum by the count of cell pixels only.
    # A cell pixel is any position where at least one valid channel is non-zero.
    cell_pixels = (
        (valid_sample != 0).any(dim=1, keepdim=True).float()
    )  # (B, 1, 1, H, W)
    cell_area = cell_pixels.sum(dim=(2, 3, 4)).clamp(min=1.0)  # (B, 1)
    mean_expression = valid_sample.sum(dim=(2, 3, 4)) / cell_area  # (B, C_max)

    masked_indices = torch.zeros(B, C_max, dtype=torch.bool, device=sample.device)
    masked_sample = sample.clone()

    for i in range(B):
        valid = (~padding_mask[i]).nonzero(as_tuple=True)[0]
        n_valid = len(valid)
        if n_valid <= min_keep:
            continue

        n_mask = max(1, int(n_valid * mask_ratio))
        # Ensure we keep at least min_keep channels unmasked
        n_mask = min(n_mask, n_valid - min_keep)
        if n_mask <= 0:
            continue

        perm = torch.randperm(n_valid, device=sample.device)[:n_mask]
        channels_to_mask = valid[perm]
        masked_indices[i, channels_to_mask] = True
        masked_sample[i, channels_to_mask] = 0.0

    return masked_sample, masked_indices, mean_expression
