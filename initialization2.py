"""
Intialization and main runtime
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotloom import Loom


# Import modules (developed by Hani and Amal)
# from physics_module import calculate_collision, update_positions
# from boundary_module import apply_boundary_conditions


def initialize_particles(M_pos, M_vel, n_particles, box_size, R):
    """Initialize small particles randomly, avoiding large particle"""
    m_pos = np.random.uniform(0, box_size, (n_particles, 2))  # uniform distribution gives equal probability anywhere in box
    m_vel = np.random.normal(0, 0.5, (n_particles, 2))  # normal distribution models thermal motion of gas particles
    
    # Remove particles overlapping with large particle
    dist = np.linalg.norm(m_pos - M_pos, axis=1)  # distance to check overlap
    while np.any(dist < R + 0.1):  # 0.1 buffer prevents particles spawning too close
        overlap = dist < R + 0.1  # Boolean mask for overlapping particles
        m_pos[overlap] = np.random.uniform(0, box_size, (np.sum(overlap), 2))  # respawn only overlapping ones
        dist = np.linalg.norm(m_pos - M_pos, axis=1)  # recalculate distances
    
    return m_pos, m_vel


def run_simulation(M, m, R, box_size, n_particles, dt, total_time, M_pos_init, M_vel_init):
    """Main simulation loop"""
    n_steps = int(total_time / dt)  # convert continuous time to discrete steps
    
    # Storage
    time = np.linspace(0, total_time, n_steps)  # evenly spaced time points for plotting
    M_pos_history = np.zeros((n_steps, 2))  # allocate for efficiency
    M_vel_history = np.zeros((n_steps, 2))
    energy = np.zeros(n_steps)  # track to verify conservation
    
    # Initialize
    M_pos, M_vel = M_pos_init.copy(), M_vel_init.copy()  # copy to avoid modifying originals
    m_pos, m_vel = initialize_particles(M_pos, M_vel, n_particles, box_size, R)
    
    for step in range(n_steps):
        # === MODULE FUNCTIONS GO HERE (developed by Hani and Amal) ===
        # collision_data = calculate_collision(M_pos, M_vel, m_pos, m_vel, M, m, R)
        # M_vel, m_vel = collision_data['updated_velocities']
        
        # PLACEHOLDER: Check collisions
        dist = np.linalg.norm(m_pos - M_pos, axis=1)  # distance from each small particle to large one
        colliding = dist <= R  # Boolean array of which particles are colliding
        # ... collision physics would update M_vel and m_vel using momentum conservation ...
        
        # Update positions (Euler)
        M_pos += M_vel * dt  # Euler: x_new = x_old + v*dt
        m_pos += m_vel * dt  # update for all small particles at once
        
        # Wall bounces (perfectly elastic)
        if M_pos[0] < R or M_pos[0] > box_size - R: M_vel[0] *= -1  # Reverse x-velocity at left/right walls
        if M_pos[1] < R or M_pos[1] > box_size - R: M_vel[1] *= -1  # Reverse y-velocity at top/bottom walls
        
        wall_x = (m_pos[:, 0] < 0) | (m_pos[:, 0] > box_size)  # Boolean mask for x-boundary violations
        wall_y = (m_pos[:, 1] < 0) | (m_pos[:, 1] > box_size)  # Boolean mask for y-boundary violations
        m_vel[wall_x, 0] *= -1  # Reverse x-component only for particles hitting vertical walls
        m_vel[wall_y, 1] *= -1  # Reverse y-component only for particles hitting horizontal walls
       
        # Store data
        M_pos_history[step] = M_pos
        M_vel_history[step] = M_vel
        energy[step] = 0.5 * M * np.sum(M_vel**2) + 0.5 * m * np.sum(m_vel**2)  # Total KE of system
        
        if step % (n_steps // 10) == 0:  # Print progress every 10%
            print(f"t={time[step]:.2f}s | speed={np.linalg.norm(M_vel):.3f} m/s")
    
    return time, M_pos_history, M_vel_history, energy


def plot_results(time, M_pos, M_vel, energy):
    """Plots and GIF animation"""

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


# Run simulation
if __name__ == "__main__":
    time, M_pos, M_vel, energy = run_simulation(
        M=1.0, m=0.001, R=0.1, box_size=10.0, n_particles=200,
        dt=0.001, total_time=5.0,
        M_pos_init=np.array([1.0, 5.0]),
        M_vel_init=np.array([10.0, 0.0])
    )
    
    print(f"\nSpeed: {np.linalg.norm(M_vel[0]):.3f} → {np.linalg.norm(M_vel[-1]):.3f} m/s")
    print(f"Energy change: {100*(energy[-1]-energy[0])/energy[0]:.4f}%")  # Should be ~0% if energy conserved
    
    plot_results(time, M_pos, M_vel, energy)
    np.savez('simulation.npz', time=time, M_pos=M_pos, M_vel=M_vel, energy=energy)  # Save for post-processing team

