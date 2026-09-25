from __future__ import annotations

import argparse
from pathlib import Path

from diffusion_functions import recommended_dt
from simulation import run_simulation
from plots import (
    concentration_profile_over_time,
    save_animation_gif,
    save_heatmap,
    save_snapshot_plot,
)
from data_utils import export_results


def interactive_inputs():
    """Preserve the original project's prompt-based workflow."""
    print("\n1D Fick's Law Diffusion Simulator\n")
    return {
        "length": float(input("Enter the length of material in meters: ")),
        "initial_concentration": float(
            input("Enter the initial concentration in the material in mol/m^3: ")
        ),
        "surface_concentration": float(
            input("Enter the surface concentration in mol/m^3: ")
        ),
        "D": float(input("Enter the diffusion coefficient in m^2/s: ")),
        "dx": float(input("Enter the spatial step size in meters: ")),
        "dt": float(input("Enter the time step size in seconds: ")),
        "total_time": float(input("Enter the total simulation time in seconds: ")),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="1D Fick's second-law diffusion simulator."
    )
    parser.add_argument(
        "--mode",
        choices=["fixed-surface", "finite-source", "interdiffusion"],
        default="fixed-surface",
        help="Initial/boundary-condition mode.",
    )
    parser.add_argument("--length", type=float)
    parser.add_argument("--initial", type=float, dest="initial_concentration")
    parser.add_argument("--surface", type=float, dest="surface_concentration")
    parser.add_argument("--D", type=float)
    parser.add_argument("--dx", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--time", type=float, dest="total_time")
    parser.add_argument(
        "--source-width",
        type=float,
        default=None,
        help="Finite-source width, or interface position for interdiffusion.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--auto-dt",
        action="store_true",
        help="Automatically choose a time step below the stability limit.",
    )
    parser.add_argument(
        "--show-animation",
        action="store_true",
        help="Open the original interactive animation after the run.",
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Save the animation as diffusion.gif.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    numeric_values = [
        args.length,
        args.initial_concentration,
        args.surface_concentration,
        args.D,
        args.dx,
        args.total_time,
    ]

    # Running with no numerical CLI inputs keeps the original interactive style.
    if all(value is None for value in numeric_values) and args.dt is None:
        values = interactive_inputs()
    else:
        required = {
            "--length": args.length,
            "--initial": args.initial_concentration,
            "--surface": args.surface_concentration,
            "--D": args.D,
            "--dx": args.dx,
            "--time": args.total_time,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("Missing required arguments for CLI mode: " + ", ".join(missing))

        dt = args.dt
        if args.auto_dt or dt is None:
            dt = recommended_dt(args.D, args.dx)
            print(f"Using automatic stable dt = {dt:.6g} s")

        values = {
            "length": args.length,
            "initial_concentration": args.initial_concentration,
            "surface_concentration": args.surface_concentration,
            "D": args.D,
            "dx": args.dx,
            "dt": dt,
            "total_time": args.total_time,
        }

    result = run_simulation(
        **values,
        save_every=args.save_every,
        mode=args.mode,
        source_width=args.source_width,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = export_results(
        result,
        output_dir,
        values["initial_concentration"],
        values["surface_concentration"],
    )
    save_snapshot_plot(
        result.x,
        result.snapshots,
        output_dir / "concentration_profiles.png",
    )
    save_heatmap(
        result.x,
        result.profiles,
        result.times,
        output_dir / "concentration_heatmap.png",
    )

    if args.save_gif:
        save_animation_gif(
            result.x,
            result.profiles,
            result.times,
            output_dir / "diffusion.gif",
        )

    print("\nSimulation complete")
    print(f"Mode:                  {result.mode}")
    print(f"Stability ratio:       {result.stability_value:.4f} (must be <= 0.5)")
    print(f"Saved profiles:        {len(result.profiles)}")
    print(f"Penetration depth:     {metadata['penetration_depth_m']:.6g} m")
    print(f"Results saved to:      {output_dir.resolve()}")

    print("\nSaved snapshots:")
    for time in sorted(result.snapshots):
        print(f"  t = {time:.4g} s")

    if args.show_animation:
        concentration_profile_over_time(
            result.x,
            result.profiles,
            result.times,
        )


if __name__ == "__main__":
    main()
