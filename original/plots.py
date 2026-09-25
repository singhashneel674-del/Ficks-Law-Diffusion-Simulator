import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button


def concentration_profile_over_time(x, profiles, times, animation_duration=10, fps=30, hold_final=2):
    profiles = np.array(profiles)
    times = np.array(times)

    # Smooth visual x-axis for blended-looking bar
    x_smooth = np.linspace(x[0], x[-1], 500)

    # Custom colormap: low concentration = black, high concentration = red
    red_cmap = LinearSegmentedColormap.from_list(
        "black_to_red",
        ["black", "darkred", "red"]
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    plt.subplots_adjust(bottom=0.25)

    # Smooth initial profile
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

    active_frames = int(animation_duration * fps)
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

    # Replay button
    button_ax = plt.axes([0.42, 0.05, 0.16, 0.075])
    replay_button = Button(button_ax, "Replay")

    def replay(event):
        ani.frame_seq = ani.new_frame_seq()
        update(0)
        fig.canvas.draw_idle()
        ani.event_source.start()

    replay_button.on_clicked(replay)
    plt.show()
