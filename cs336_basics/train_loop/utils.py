import math


def cosine_annealing(t: int, alpha_max: float, alpha_min: float, t_w: int, t_c: int) -> float:
    """
    Cosine annealing learning rate scheduling
    Args:
        t (int): current iteration
        alpha_max (float): maximum learning rate
        alpha_min (float): minimum (final) learning rate
        t_w (int): number of warmup iterations
        t_c (int): final iteration of cosine annealing
    Return:
        lr_t (float): learning rate at iteration t
    """
    if t < t_w:  # warmup
        return (alpha_max / t_w) * t
    elif t <= t_c:  # cosine annealing
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi / (t_c - t_w) * (t - t_w)))
    else:
        return alpha_min
