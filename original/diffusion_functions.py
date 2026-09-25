import numpy as np


def updated_concentration(C, D, dt, dx):
    """
    Update the concentration using 1D fick's second law of diffusion.
    """
    C_new = C.copy()

    for i in range(1, len(C)-1):
        second_derivative = (C[i+1] - 2*C[i] + C[i-1]) / dx**2
        C_new[i] = C[i] + D * second_derivative * dt
    return C_new


def check_stability(D, dt, dx):
    """
    check if dt is small enough for for dx spacing and diffusion coefficient D
    """
    stability_limit = D*dt/dx**2

    if stability_limit > 0.5:
        print("Warning: The simulation may be unstable. Consider reducing dt or increasing dx.")
    else:
        print("stable simulation")
    return stability_limit
