"""
Physics Module for N-Body Simulation
Developed by: Hanieh (Module Developer)
Handles collision detection, collision response, and boundary conditions
"""

import numpy as np

def check_sphere_particle_collisions(M_pos, m_pos, R):
    """
    Check which particles are colliding with the large sphere
    
    Parameters:
    - M_pos: array [x, y] position of large particle
    - m_pos: array (n_particles, 2) positions of small particles
    - R: radius of large particle
    
    Returns:
    - colliding: boolean array indicating which particles are colliding
    """
    dist = np.linalg.norm(m_pos - M_pos, axis=1)
    return dist <= R

def handle_collision_response(M_pos, M_vel, m_pos, m_vel, M, m, R, colliding):
    """
    Handle elastic collisions between sphere and particles
    Uses conservation of momentum and energy
    
    For elastic collision in 2D:
    - Decompose velocities into normal (along collision line) and tangential components
    - Apply 1D collision formula to normal components:
        vb' = ((M-m)*vb + 2*m*vp) / (M+m)
        vp' = ((m-M)*vp + 2*M*vb) / (M+m)
    - Tangential components remain unchanged
    
    Parameters:
    - M_pos: array [x, y] position of large particle
    - M_vel: array [vx, vy] velocity of large particle
    - m_pos: array (n_particles, 2) positions of small particles
    - m_vel: array (n_particles, 2) velocities of small particles
    - M: mass of large particle
    - m: mass of each small particle
    - R: radius of large particle
    - colliding: boolean array of which particles are colliding
    
    Returns:
    - M_vel_new: updated velocity of large particle
    - m_vel_new: updated velocities of small particles
    """
    M_vel_new = M_vel.copy()
    m_vel_new = m_vel.copy()
    
    if not np.any(colliding):
        return M_vel_new, m_vel_new
    
    # Get indices of colliding particles
    collision_indices = np.where(colliding)[0]
    
    for idx in collision_indices:
        # Calculate collision normal (from sphere center to particle)
        dx = m_pos[idx, 0] - M_pos[0]
        dy = m_pos[idx, 1] - M_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 1e-10:  # Avoid division by zero
            continue
        
        # Normal vector (unit vector pointing from M to particle)
        nx = dx / distance
        ny = dy / distance
        
        # Project velocities onto collision normal
        vb_normal = M_vel_new[0] * nx + M_vel_new[1] * ny
        vp_normal = m_vel_new[idx, 0] * nx + m_vel_new[idx, 1] * ny
        
        # Tangential components (perpendicular to normal)
        vb_tangent_x = M_vel_new[0] - vb_normal * nx
        vb_tangent_y = M_vel_new[1] - vb_normal * ny
        vp_tangent_x = m_vel_new[idx, 0] - vp_normal * nx
        vp_tangent_y = m_vel_new[idx, 1] - vp_normal * ny
        
        # Apply 1D elastic collision formula along normal direction
        # From conservation of momentum and energy:
        vb_normal_new = ((M - m) * vb_normal + 2 * m * vp_normal) / (M + m)
        vp_normal_new = ((m - M) * vp_normal + 2 * M * vb_normal) / (M + m)
        
        # Combine normal and tangential components to get final velocities
        M_vel_new[0] = vb_normal_new * nx + vb_tangent_x
        M_vel_new[1] = vb_normal_new * ny + vb_tangent_y
        m_vel_new[idx, 0] = vp_normal_new * nx + vp_tangent_x
        m_vel_new[idx, 1] = vp_normal_new * ny + vp_tangent_y
        
        # Separate particles slightly to prevent overlap in next step
        overlap = R - distance
        if overlap > 0:
            m_pos[idx, 0] += nx * (overlap + 0.01)
            m_pos[idx, 1] += ny * (overlap + 0.01)
    
    return M_vel_new, m_vel_new

def apply_wall_bounces_large_particle(M_pos, M_vel, box_size, R):
    """
    Apply boundary conditions for large particle (elastic wall collisions)
    
    Parameters:
    - M_pos: array [x, y] position of large particle
    - M_vel: array [vx, vy] velocity of large particle
    - box_size: size of simulation box
    - R: radius of large particle
    
    Returns:
    - M_vel_new: updated velocity after wall bounces
    """
    M_vel_new = M_vel.copy()
    
    # Check left/right walls (account for radius)
    if M_pos[0] < R or M_pos[0] > box_size - R:
        M_vel_new[0] *= -1
    
    # Check top/bottom walls (account for radius)
    if M_pos[1] < R or M_pos[1] > box_size - R:
        M_vel_new[1] *= -1
    
    return M_vel_new

def apply_wall_bounces_small_particles(m_pos, m_vel, box_size):
    """
    Apply boundary conditions for small particles (elastic wall collisions)
    
    Parameters:
    - m_pos: array (n_particles, 2) positions of small particles
    - m_vel: array (n_particles, 2) velocities of small particles
    - box_size: size of simulation box
    
    Returns:
    - m_vel_new: updated velocities after wall bounces
    """
    m_vel_new = m_vel.copy()
    
    # Check x boundaries (left/right walls)
    wall_x = (m_pos[:, 0] < 0) | (m_pos[:, 0] > box_size)
    m_vel_new[wall_x, 0] *= -1
    
    # Check y boundaries (top/bottom walls)
    wall_y = (m_pos[:, 1] < 0) | (m_pos[:, 1] > box_size)
    m_vel_new[wall_y, 1] *= -1
    
    return m_vel_new

def calculate_total_energy(M_vel, m_vel, M, m):
    """
    Calculate total kinetic energy of the system
    
    Parameters:
    - M_vel: velocity of large particle
    - m_vel: velocities of small particles
    - M: mass of large particle
    - m: mass of each small particle
    
    Returns:
    - total_energy: total kinetic energy
    """
    # KE of large particle
    KE_large = 0.5 * M * np.sum(M_vel**2)
    
    # KE of all small particles
    KE_small = 0.5 * m * np.sum(m_vel**2)
    
    return KE_large + KE_small

def calculate_total_momentum(M_vel, m_vel, M, m):
    """
    Calculate total momentum of the system
    
    Parameters:
    - M_vel: velocity of large particle
    - m_vel: velocities of small particles
    - M: mass of large particle
    - m: mass of each small particle
    
    Returns:
    - px, py: x and y components of total momentum
    """
    # Momentum of large particle
    p_large = M * M_vel
    
    # Momentum of all small particles
    p_small = m * np.sum(m_vel, axis=0)
    
    px = p_large[0] + p_small[0]
    py = p_large[1] + p_small[1]
    
    return px, py


# ============= TESTING =============
if __name__ == "__main__":
    print("Testing Physics Module...\n")
    
    # Test 1: Simple head-on collision
    print("Test 1: Head-on collision")
    M, m = 1.0, 0.001
    M_pos = np.array([5.0, 5.0])
    M_vel = np.array([1.0, 0.0])
    m_pos = np.array([[5.5, 5.0]])  # Particle directly in front
    m_vel = np.array([[0.0, 0.0]])
    R = 0.6
    
    print(f"Before: M_vel = {M_vel}, m_vel = {m_vel[0]}")
    
    colliding = check_sphere_particle_collisions(M_pos, m_pos, R)
    print(f"Collision detected: {colliding[0]}")
    
    M_vel_new, m_vel_new = handle_collision_response(
        M_pos, M_vel, m_pos, m_vel, M, m, R, colliding
    )
    print(f"After:  M_vel = {M_vel_new}, m_vel = {m_vel_new[0]}")
    
    # Check momentum conservation
    p_before = M * M_vel + m * m_vel[0]
    p_after = M * M_vel_new + m * m_vel_new[0]
    print(f"Momentum before: {p_before}")
    print(f"Momentum after:  {p_after}")
    print(f"Momentum conserved: {np.allclose(p_before, p_after)}\n")
    
    # Test 2: Wall bounce
    print("Test 2: Wall bounce")
    M_pos = np.array([0.05, 5.0])
    M_vel = np.array([-1.0, 0.0])
    R = 0.2
    box_size = 10.0
    
    print(f"Before wall: M_vel = {M_vel}")
    M_vel_new = apply_wall_bounces_large_particle(M_pos, M_vel, box_size, R)
    print(f"After wall:  M_vel = {M_vel_new}")
    print(f"X-velocity reversed: {M_vel_new[0] == -M_vel[0]}\n")
    
    # Test 3: Energy conservation
    print("Test 3: Energy conservation in collision")
    M_vel = np.array([2.0, 0.0])
    m_vel = np.array([[0.0, 0.0]])
    
    E_before = calculate_total_energy(M_vel, m_vel, M, m)
    
    M_pos = np.array([5.0, 5.0])
    m_pos = np.array([[5.5, 5.0]])
    colliding = check_sphere_particle_collisions(M_pos, m_pos, R)
    M_vel_new, m_vel_new = handle_collision_response(
        M_pos, M_vel, m_pos, m_vel, M, m, R, colliding
    )
    
    E_after = calculate_total_energy(M_vel_new, m_vel_new, M, m)
    
    print(f"Energy before: {E_before:.6f}")
    print(f"Energy after:  {E_after:.6f}")
    print(f"Energy conserved: {np.allclose(E_before, E_after)}")
    print(f"Energy change: {abs(E_after - E_before):.2e}")
