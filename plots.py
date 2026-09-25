import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button


def concentration_profile_over_time(
    x,
    profiles,
    times,
    animation_duration=10,
    fps=30,
    hold_final=2,
    show=True,
):
    """
    Interactive animated concentration bar.

    This preserves the visualization approach from the original project:
    concentration is represented with a black-to-red bar, a time display,
    interpolation between saved profiles, and a replay button.
    """
    profiles = np.array(profiles)
    times = np.array(times)

    x_smooth = np.linspace(x[0], x[-1], 500)

    red_cmap = LinearSegmentedColormap.from_list(
        "black_to_red",
        ["black", "darkred", "red"]
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    plt.subplots_adjust(bottom=0.25)

    initial_profile = np.interp(x_smooth, x, profiles[0])

    bar = ax.imshow(
        [initial_profile],
        aspect="auto",
        cmap=red_cmap,
        interpolation="bicubic",
        extent=[x[0], x[-1], 0, 1],
        vmin=np.min(profiles),
        vmax=np.max(profiles)
    )

    cbar = plt.colorbar(bar, ax=ax)
    cbar.set_label("Concentration (mol/m^3)")

    ax.set_xlabel("Position (m)")
    ax.set_yticks([])
    ax.set_title("1D Diffusion Through Material")

    timer_text = ax.text(
        0.98, 1.15,
        f"Time = {times[0]:.2f} s",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=12
    )

    active_frames = max(2, int(animation_duration * fps))
    hold_frames = int(hold_final * fps)
    total_frames = active_frames + hold_frames
    interval = 1000 / fps

    def get_frame_data(frame):
        if frame >= active_frames:
            profile_now = profiles[-1]
            time_now = times[-1]
        else:
            progress = frame / (active_frames - 1) * (len(profiles) - 1)

            i0 = int(np.floor(progress))
            i1 = min(i0 + 1, len(profiles) - 1)
            alpha = progress - i0

            profile_now = (1 - alpha) * profiles[i0] + alpha * profiles[i1]
            time_now = (1 - alpha) * times[i0] + alpha * times[i1]

        profile_smooth = np.interp(x_smooth, x, profile_now)
        return profile_smooth, time_now

    def update(frame):
        profile_smooth, time_now = get_frame_data(frame)
        bar.set_data([profile_smooth])
        timer_text.set_text(f"Time = {time_now:.2f} s")
        return bar, timer_text

    ani = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=interval,
        blit=False,
        repeat=True,
        repeat_delay=100
    )

    button_ax = plt.axes([0.42, 0.05, 0.16, 0.075])
    replay_button = Button(button_ax, "Replay")

    def replay(event):
        ani.frame_seq = ani.new_frame_seq()
        update(0)
        fig.canvas.draw_idle()
        ani.event_source.start()

    replay_button.on_clicked(replay)

    if show:
        plt.show()

    return fig, ani


def save_snapshot_plot(x, snapshots, output_path):
    """Save concentration-vs-position curves for the five main snapshots."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for time in sorted(snapshots):
        ax.plot(x, snapshots[time], label=f"t = {time:.3g} s")

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Concentration (mol/m³)")
    ax.set_title("1D Diffusion Concentration Profiles")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_heatmap(x, profiles, times, output_path):
    """Save concentration as a position-time heatmap."""
    profiles = np.asarray(profiles)
    times = np.asarray(times)

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(
        profiles,
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], times[0], times[-1]],
    )
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Diffusion Through Time")
    fig.colorbar(image, ax=ax, label="Concentration (mol/m³)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_animation_gif(x, profiles, times, output_path, fps=20):
    """Save the animated concentration bar as a GIF."""
    fig, ani = concentration_profile_over_time(
        x,
        profiles,
        times,
        animation_duration=6,
        fps=fps,
        hold_final=1,
        show=False,
    )
    ani.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
