import numpy as np
import matplotlib.pyplot as plt
from matplotloom import Loom

def plot_results(time, M_pos, M_vel, energy):
    """
    Plots and GIF animation
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # speed magnitude
    speed = np.linalg.norm(M_vel, axis=1)
    axes[0].plot(time, speed)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].grid(True)

    # speed^2 as a proxy for drag magnitude
    drag_mag_prop = speed**2  
    axes[1].plot(time, drag_mag_prop)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed Squared (m^2/s^2)")
    axes[1].grid(True)

    # energy
    axes[2].plot(time, energy)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Energy (J)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.close(fig)

    n_steps = len(time)
    with Loom("ball_trajectory.gif", fps=30) as loom:
        for i in range(n_steps):
            fig, ax = plt.subplots(figsize=(5,5))

            x, y = M_pos[i]
            ax.scatter(x, y, s=50)

            # ax.plot(M_pos[:i,0], M_pos[:i,1], alpha=0.3)

            ax.set_xlim(np.min(M_pos[:,0]) - 0.5, np.max(M_pos[:,0]) + 0.5)
            ax.set_ylim(np.min(M_pos[:,1]) - 0.5, np.max(M_pos[:,1]) + 0.5)
            ax.set_aspect("equal")
            ax.set_title(f"t = {time[i]:.3f} s")
            loom.save_frame(fig)
            plt.close(fig)

