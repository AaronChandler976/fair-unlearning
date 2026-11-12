
import torch

from src.objectives.derivatives import (
    compute_gradient,
    compute_hessian,
)


def newton_unlearn(
    theta_original,
    X_remaining,
    y_remaining,
    sensitive_remaining,
    lambda_reg=1e-4,
    gamma=1.0,
    damping=0.0,
):
    """
    Approximate the model trained on the remaining dataset
    using one Newton update from the original model.
    """

    gradient = compute_gradient(
        theta_original,
        X_remaining,
        y_remaining,
        sensitive_remaining,
        lambda_reg=lambda_reg,
        gamma=gamma,
    )

    hessian = compute_hessian(
        theta_original,
        X_remaining,
        y_remaining,
        sensitive_remaining,
        lambda_reg=lambda_reg,
        gamma=gamma,
    )

    identity = torch.eye(
        hessian.shape[0],
        dtype=hessian.dtype,
        device=hessian.device,
    )

    # Optional damping (default 0, as in the paper). The L2 term already
    # makes the Hessian positive definite, so no damping is needed.
    damped_hessian = hessian + damping * identity

    # Solve H * step = gradient instead of computing H inverse.
    step = torch.linalg.solve(damped_hessian, gradient)

    theta_unlearned = theta_original.detach() - step

    return theta_unlearned, {
        "gradient": gradient,
        "hessian": hessian,
        "step": step,
    }
