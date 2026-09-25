import numpy as np
import pytest

from diffusion_functions import (
    updated_concentration,
    check_stability,
    recommended_dt,
)
from simulation import run_simulation


def test_original_update_matches_finite_difference_equation():
    C = np.array([1.0, 0.5, 0.0])
    result = updated_concentration(C, D=0.1, dt=0.01, dx=0.1)

    second_derivative = (0.0 - 2 * 0.5 + 1.0) / 0.1**2
    expected_center = 0.5 + 0.1 * second_derivative * 0.01
    assert np.isclose(result[1], expected_center)


def test_stability_boundary():
    ratio = check_stability(D=1.0, dt=0.005, dx=0.1)
    assert np.isclose(ratio, 0.5)


def test_recommended_dt_is_stable():
    dt = recommended_dt(D=1.0, dx=0.1)
    ratio = 1.0 * dt / 0.1**2
    assert ratio < 0.5


def test_unstable_run_is_rejected():
    with pytest.raises(ValueError):
        run_simulation(
            length=1.0,
            initial_concentration=0.0,
            surface_concentration=1.0,
            D=1.0,
            dx=0.1,
            dt=0.01,
            total_time=0.1,
            verbose=False,
        )


def test_fixed_surface_boundaries_remain_fixed():
    result = run_simulation(
        length=1.0,
        initial_concentration=0.0,
        surface_concentration=1.0,
        D=0.1,
        dx=0.1,
        dt=0.04,
        total_time=0.2,
        save_every=1,
        verbose=False,
    )
    assert np.allclose(result.profiles[:, 0], 1.0)
    assert np.allclose(result.profiles[:, -1], 0.0)


def test_profiles_are_not_duplicated_by_snapshot_loop():
    result = run_simulation(
        length=1.0,
        initial_concentration=0.0,
        surface_concentration=1.0,
        D=0.1,
        dx=0.1,
        dt=0.04,
        total_time=0.2,
        save_every=1,
        verbose=False,
    )
    expected_steps = int(0.2 / 0.04) + 1
    assert len(result.profiles) == expected_steps


def test_finite_source_runs_with_closed_boundaries():
    result = run_simulation(
        length=1.0,
        initial_concentration=0.0,
        surface_concentration=1.0,
        D=0.1,
        dx=0.05,
        dt=0.01,
        total_time=0.1,
        mode="finite-source",
        source_width=0.2,
        save_every=1,
        verbose=False,
    )
    assert result.mode == "finite-source"
    assert np.all(np.isfinite(result.profiles))


def test_interdiffusion_interface_smooths():
    result = run_simulation(
        length=1.0,
        initial_concentration=0.0,
        surface_concentration=1.0,
        D=0.1,
        dx=0.05,
        dt=0.01,
        total_time=0.1,
        mode="interdiffusion",
        source_width=0.5,
        save_every=1,
        verbose=False,
    )
    initial_jump = np.max(np.abs(np.diff(result.profiles[0])))
    final_jump = np.max(np.abs(np.diff(result.profiles[-1])))
    assert final_jump < initial_jump
