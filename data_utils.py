from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np


def total_concentration(x, profile):
    """Integral of concentration over position."""
    return float(np.trapezoid(profile, x))


def final_flux(x, profile, D):
    """Calculate Fick's first-law flux, J = -D dC/dx."""
    return -D * np.gradient(profile, x)


def penetration_depth(x, profile, background, surface, fraction=0.01):
    """
    Estimate the furthest position whose concentration change is at least
    'fraction' of the surface-to-background concentration difference.
    """
    scale = abs(surface - background)
    if scale == 0:
        return 0.0

    changed = np.where(abs(profile - background) >= fraction * scale)[0]
    return float(x[changed[-1]]) if len(changed) else 0.0


def export_results(result, output_dir, initial_concentration, surface_concentration):
    """Export simulation profiles, snapshots, flux, and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles_path = output_dir / "concentration_profiles.csv"
    with profiles_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "position_m", "concentration_mol_m3"])
        for time, profile in zip(result.times, result.profiles):
            for x, c in zip(result.x, profile):
                writer.writerow([time, x, c])

    snapshots_path = output_dir / "snapshots.csv"
    with snapshots_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "position_m", "concentration_mol_m3"])
        for time in sorted(result.snapshots):
            for x, c in zip(result.x, result.snapshots[time]):
                writer.writerow([time, x, c])

    flux = final_flux(result.x, result.profiles[-1], result.D)
    flux_path = output_dir / "final_flux.csv"
    with flux_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["position_m", "flux_mol_m2_s"])
        for x, j in zip(result.x, flux):
            writer.writerow([x, j])

    totals = [total_concentration(result.x, p) for p in result.profiles]
    metadata = {
        "mode": result.mode,
        "D_m2_s": result.D,
        "dx_m": result.dx,
        "dt_s": result.dt,
        "total_time_s": result.total_time,
        "stability_ratio": result.stability_value,
        "saved_profiles": len(result.profiles),
        "initial_integrated_concentration": totals[0],
        "final_integrated_concentration": totals[-1],
        "penetration_depth_m": penetration_depth(
            result.x,
            result.profiles[-1],
            initial_concentration,
            surface_concentration,
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata
