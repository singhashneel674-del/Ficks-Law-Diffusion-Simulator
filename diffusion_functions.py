import numpy as np


def updated_concentration(C, D, dt, dx):
    """
    Update the concentration using 1D Fick's second law of diffusion.

    This is the same explicit finite-difference update used in the original
    project code.
    """
    C_new = C.copy()

    for i in range(1, len(C)-1):
        second_derivative = (C[i+1] - 2*C[i] + C[i-1]) / dx**2
        C_new[i] = C[i] + D * second_derivative * dt
    return C_new


def check_stability(D, dt, dx):
    """
    Check whether dt is small enough for dx and diffusion coefficient D.

    The explicit finite-difference scheme is stable when:
        D * dt / dx**2 <= 0.5
    """
    stability_limit = D*dt/dx**2

    if stability_limit > 0.5:
        print("Warning: The simulation may be unstable. Consider reducing dt or increasing dx.")
    else:
        print("stable simulation")
    return stability_limit


def maximum_stable_dt(D, dx):
    """Return the theoretical maximum stable time step for the explicit solver."""
    if D <= 0 or dx <= 0:
        raise ValueError("D and dx must be positive.")
    return dx**2 / (2 * D)


def recommended_dt(D, dx, safety_factor=0.9):
    """Return a time step below the stability limit."""
    if not 0 < safety_factor <= 1:
        raise ValueError("safety_factor must be in the interval (0, 1].")
    return safety_factor * maximum_stable_dt(D, dx)
