from matplotloom import Loom

def plot_results(time, M_pos, M_vel, energy):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    speed = np.linalg.norm(M_vel, axis=1)
    axes[0].plot(time, speed)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Speed (m/s)')
    axes[0].grid(True)

    drag_mag_prop = speed ** 2
    axes[1].plot(time, drag_mag_prop)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Speed Squared (m^2/s^2)')
    axes[1].grid(True)

    axes[2].plot(time, energy)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Energy (J)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()

    with Loom("ball_trajectory.gif", fps=30) as loom:
        for time_s in time:
            fig, ax = plt.subplots()
            x = M_pos[time_s, 0]
            y = M_pos[time_s, 1]
            ax.plot(x, y)
            # test with bounds per frame
            # add color bar for number of interactions at step!
            # maybe add a version with trace
            loom.save_frame(fig)


