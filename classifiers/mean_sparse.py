import re

import torch


DEFAULT_MEANSPARSE_STAT_EPS = 1e-6
MEANSPARSE_MODE_NEAR_MEAN = "near_mean"
MEANSPARSE_MODE_AWAY_FROM_MEAN = "away_from_mean"
MEANSPARSE_MODES = (
    MEANSPARSE_MODE_NEAR_MEAN,
    MEANSPARSE_MODE_AWAY_FROM_MEAN,
)
DEFAULT_MEANSPARSE_MODE = MEANSPARSE_MODE_AWAY_FROM_MEAN


def format_cache_value(value):
    text = str(value)
    for old, new in (("/", "_"), (" ", ""), (".", "p"), ("-", "m")):
        text = text.replace(old, new)
    return text


def is_meansparse_enabled(alpha):
    return float(alpha) > 0.0


def validate_meansparse_mode(mode):
    if mode not in MEANSPARSE_MODES:
        raise ValueError(f"Unsupported MeanSparse mode '{mode}'. Expected one of {MEANSPARSE_MODES}.")
    return mode


def _build_meansparse_mode_tag(mode):
    mode = validate_meansparse_mode(mode)
    if mode == MEANSPARSE_MODE_NEAR_MEAN:
        return "msmnear"
    return "msmaway"


def build_meansparse_tag(
    alpha,
    beta,
    stat_eps=DEFAULT_MEANSPARSE_STAT_EPS,
    separator="_",
    mode=DEFAULT_MEANSPARSE_MODE,
):
    if not is_meansparse_enabled(alpha):
        return ""

    parts = [
        f"msa{format_cache_value(alpha)}",
        f"msb{format_cache_value(beta)}",
        _build_meansparse_mode_tag(mode),
    ]
    if float(stat_eps) != DEFAULT_MEANSPARSE_STAT_EPS:
        parts.append(f"mseps{format_cache_value(stat_eps)}")
    return separator + separator.join(parts)


def strip_meansparse_tag(text):
    text = re.sub(r"_msa[^_]+_msb[^_]+(?:_msm[^_]+)?(?:_mseps[^_]+)?", "", text)
    text = re.sub(r"-msa[^-]+-msb[^-]+(?:-msm[^-]+)?(?:-mseps[^-]+)?", "", text)
    return text


def apply_mean_centered_soft_threshold(
    x,
    mean,
    std,
    alpha,
    beta,
    stat_eps=DEFAULT_MEANSPARSE_STAT_EPS,
    mode=DEFAULT_MEANSPARSE_MODE,
):
    if not is_meansparse_enabled(alpha):
        return x
    mode = validate_meansparse_mode(mode)

    view_shape = (1,) * (x.dim() - 1) + (-1,)
    x_float = x.float()
    mean = mean.to(device=x.device, dtype=torch.float32).view(view_shape)
    std = std.to(device=x.device, dtype=torch.float32).clamp_min(float(stat_eps)).view(view_shape)
    diff = x_float - mean
    diff_abs = diff.abs()
    radius = diff_abs / std
    gate = torch.sigmoid(float(beta) * (radius - float(alpha)))
    if mode == MEANSPARSE_MODE_NEAR_MEAN:
        output = mean + gate * diff
    else:
        boundary = float(alpha) * std
        boundary_aligned_abs = boundary + gate * (diff_abs - boundary)
        output = mean + diff.sign() * boundary_aligned_abs
    return output.to(dtype=x.dtype)
