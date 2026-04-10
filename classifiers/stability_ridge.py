import torch

from classifiers.mean_sparse import format_cache_value


DEFAULT_STABILITY_RIDGE_STAT_EPS = 1e-6


def is_stability_ridge_enabled(gamma):
    return float(gamma) > 0.0


def build_stability_ridge_tag(gamma, stat_eps=DEFAULT_STABILITY_RIDGE_STAT_EPS, separator="_"):
    if not is_stability_ridge_enabled(gamma):
        return ""

    parts = [f"srg{format_cache_value(gamma)}"]
    if float(stat_eps) != DEFAULT_STABILITY_RIDGE_STAT_EPS:
        parts.append(f"sreps{format_cache_value(stat_eps)}")
    return separator + separator.join(parts)


def compute_stability_ridge_prior(projected_sum, projected_sum_sq, projected_abs_sum, sample_count, gamma, stat_eps):
    if not is_stability_ridge_enabled(gamma):
        return None
    if sample_count <= 0:
        raise ValueError("Stability-aware ridge requires at least one projected feature sample.")

    mean = projected_sum / float(sample_count)
    second_moment = projected_sum_sq / float(sample_count)
    mean_abs = projected_abs_sum / float(sample_count)
    variance = (second_moment - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(float(stat_eps))
    stability = mean_abs.clamp_min(float(stat_eps)) / std
    prior = stability.pow(-float(gamma))
    prior = prior / prior.mean().clamp_min(float(stat_eps))
    return prior.float()


def solve_ridge_system(g_matrix, q_matrix, ridge, diagonal_prior=None):
    system = g_matrix.clone()
    if diagonal_prior is None:
        system.diagonal().add_(float(ridge))
    else:
        system.diagonal().add_(float(ridge) * diagonal_prior.to(device=system.device, dtype=system.dtype))
    return torch.linalg.solve(system, q_matrix)
