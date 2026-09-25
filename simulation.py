from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from diffusion_functions import updated_concentration, check_stability


@dataclass
class SimulationResult:
    x: np.ndarray
    profiles: np.ndarray
    times: np.ndarray
    snapshots: dict[float, np.ndarray]
    D: float
    dx: float
    dt: float
    total_time: float
    mode: str

    @property
    def stability_value(self):
        return self.D * self.dt / self.dx**2


def _validate_inputs(length, initial_concentration, surface_concentration, D, dx, dt, total_time):
    if length <= 0:
        raise ValueError("Length must be positive.")
    if D <= 0:
        raise ValueError("D must be positive.")
    if dx <= 0 or dt <= 0 or total_time <= 0:
        raise ValueError("dx, dt, and total_time must be positive.")
    if dx > length:
        raise ValueError("dx should not be larger than the material length.")


def _initial_condition(x, initial_concentration, surface_concentration, mode, source_width):
    if mode == "fixed-surface":
        C = np.ones(len(x)) * initial_concentration
        C[0] = surface_concentration
        return C

    if mode == "finite-source":
        if source_width is None:
            source_width = 0.2 * x[-1]
        if not 0 < source_width < x[-1]:
            raise ValueError("source_width must be between 0 and the material length.")
        C = np.ones(len(x)) * initial_concentration
        C[x <= source_width] = surface_concentration
        return C

    if mode == "interdiffusion":
        if source_width is None:
            source_width = 0.5 * x[-1]
        if not 0 < source_width < x[-1]:
            raise ValueError("For interdiffusion, source_width is used as the interface position.")
        return np.where(x <= source_width, surface_concentration, initial_concentration).astype(float)

    raise ValueError(f"Unknown simulation mode: {mode}")


def _apply_boundaries(C, mode, surface_concentration, initial_concentration):
    if mode == "fixed-surface":
        C[0] = surface_concentration
        C[-1] = initial_concentration
    else:
        # Zero-gradient outer boundaries for a closed finite domain.
        C[0] = C[1]
        C[-1] = C[-2]


def run_simulation(
    length,
    initial_concentration,
    surface_concentration,
    D,
    dx,
    dt,
    total_time,
    save_every=10,
    mode="fixed-surface",
    source_width=None,
    verbose=True,
):
    """
    Run a 1D Fick's-second-law simulation using the original finite-difference
    update function from diffusion_functions.py.
    """
    _validate_inputs(length, initial_concentration, surface_concentration, D, dx, dt, total_time)

    if save_every < 1:
        raise ValueError("save_every must be at least 1.")

    stability_value = check_stability(D, dt, dx) if verbose else D * dt / dx**2
    if stability_value > 0.5:
        max_dt = dx**2 / (2 * D)
        raise ValueError(
            f"Unstable simulation: D*dt/dx^2 = {stability_value:.4f}. "
            f"Use dt <= {max_dt:.6g} s for the selected D and dx."
        )

    x = np.arange(0, length + dx, dx)
    C = _initial_condition(x, initial_concentration, surface_concentration, mode, source_width)
    _apply_boundaries(C, mode, surface_concentration, initial_concentration)

    number_of_steps = int(total_time / dt)
    target_snapshot_times = np.array(
        [0, total_time/4, total_time/2, 3*total_time/4, total_time],
        dtype=float,
    )

    snapshots = {}
    profiles = []
    times = []

    # Save each profile once per requested interval. In the original main.py,
    # this was inside the snapshot loop, so each saved profile was duplicated.
    for step in range(number_of_steps + 1):
        current_time = step * dt

        if step % save_every == 0 or step == number_of_steps:
            profiles.append(C.copy())
            times.append(current_time)

        for snap_time in target_snapshot_times:
            if snap_time not in snapshots and abs(current_time - snap_time) <= dt / 2:
                snapshots[float(snap_time)] = C.copy()

        if step == number_of_steps:
            break

        C = updated_concentration(C, D, dt, dx)
        _apply_boundaries(C, mode, surface_concentration, initial_concentration)

    # Guarantee the final profile is available as a snapshot.
    snapshots[float(total_time)] = C.copy()

    return SimulationResult(
        x=x,
        profiles=np.asarray(profiles),
        times=np.asarray(times),
        snapshots=snapshots,
        D=D,
        dx=dx,
        dt=dt,
        total_time=total_time,
        mode=mode,
    )
