"""
Quick integration test to verify physics module works with team's code
"""

import numpy as np
import sys

# Import your physics module
try:
    from nbody_physics import (
        check_sphere_particle_collisions,
        handle_collision_response,
        apply_wall_bounces_large_particle,
        apply_wall_bounces_small_particles,
        calculate_total_energy,
        calculate_total_momentum
    )
    print("✓ Physics module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import physics module: {e}")
    sys.exit(1)

def test_collision_conservation():
    """Test that collisions conserve momentum and energy"""
    print("\n=== Testing Collision Conservation ===")
    
    # Setup
    M, m = 1.0, 0.001
    M_pos = np.array([5.0, 5.0])
    M_vel = np.array([2.0, 0.0])
    m_pos = np.array([[5.5, 5.0]])
    m_vel = np.array([[0.0, 0.0]])
    R = 0.6
    
    # Before collision
    E_before = calculate_total_energy(M_vel, m_vel, M, m)
    px_before, py_before = calculate_total_momentum(M_vel, m_vel, M, m)
    
    # Apply collision
    colliding = check_sphere_particle_collisions(M_pos, m_pos, R)
    M_vel_new, m_vel_new = handle_collision_response(
        M_pos, M_vel, m_pos, m_vel, M, m, R, colliding
    )
    
    # After collision
    E_after = calculate_total_energy(M_vel_new, m_vel_new, M, m)
    px_after, py_after = calculate_total_momentum(M_vel_new, m_vel_new, M, m)
    
    # Check conservation
    energy_conserved = np.isclose(E_before, E_after, rtol=1e-10)
    momentum_x_conserved = np.isclose(px_before, px_after, rtol=1e-10)
    momentum_y_conserved = np.isclose(py_before, py_after, rtol=1e-10)
    
    print(f"Energy: {E_before:.6f} → {E_after:.6f}")
    print(f"  Conserved: {'✓' if energy_conserved else '✗'}")
    print(f"Momentum X: {px_before:.6f} → {px_after:.6f}")
    print(f"  Conserved: {'✓' if momentum_x_conserved else '✗'}")
    print(f"Momentum Y: {py_before:.6f} → {py_after:.6f}")
    print(f"  Conserved: {'✓' if momentum_y_conserved else '✗'}")
    
    return energy_conserved and momentum_x_conserved and momentum_y_conserved

def test_wall_bounces():
    """Test that wall bounces work correctly"""
    print("\n=== Testing Wall Bounces ===")
    
    # Test large particle
    M_pos = np.array([0.05, 5.0])
    M_vel = np.array([-1.0, 0.5])
    R = 0.2
    box_size = 10.0
    
    M_vel_new = apply_wall_bounces_large_particle(M_pos, M_vel, box_size, R)
    
    x_reversed = (M_vel_new[0] == -M_vel[0])
    y_unchanged = (M_vel_new[1] == M_vel[1])
    
    print(f"Large particle near left wall:")
    print(f"  Velocity: {M_vel} → {M_vel_new}")
    print(f"  X reversed: {'✓' if x_reversed else '✗'}")
    print(f"  Y unchanged: {'✓' if y_unchanged else '✗'}")
    
    # Test small particles
    m_pos = np.array([[0.1, 5.0], [9.9, 5.0], [5.0, 0.1]])
    m_vel = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0]])
    
    m_vel_new = apply_wall_bounces_small_particles(m_pos, m_vel, box_size)
    
    all_reversed = (
        m_vel_new[0, 0] == -m_vel[0, 0] and
        m_vel_new[1, 0] == -m_vel[1, 0] and
        m_vel_new[2, 1] == -m_vel[2, 1]
    )
    
    print(f"Small particles near walls:")
    print(f"  All velocities reversed correctly: {'✓' if all_reversed else '✗'}")
    
    return x_reversed and y_unchanged and all_reversed

def test_no_collision_case():
    """Test that no collision means no velocity change"""
    print("\n=== Testing No Collision Case ===")
    
    M, m = 1.0, 0.001
    M_pos = np.array([5.0, 5.0])
    M_vel = np.array([1.0, 0.0])
    m_pos = np.array([[8.0, 5.0], [2.0, 2.0]])  # Far away
    m_vel = np.array([[0.0, 0.0], [0.5, 0.5]])
    R = 0.5
    
    colliding = check_sphere_particle_collisions(M_pos, m_pos, R)
    M_vel_new, m_vel_new = handle_collision_response(
        M_pos, M_vel, m_pos, m_vel, M, m, R, colliding
    )
    
    velocities_unchanged = (
        np.allclose(M_vel, M_vel_new) and
        np.allclose(m_vel, m_vel_new)
    )
    
    print(f"No collisions detected: {not np.any(colliding)}")
    print(f"Velocities unchanged: {'✓' if velocities_unchanged else '✗'}")
    
    return velocities_unchanged

def run_mini_simulation():
    """Run a tiny simulation to check integration"""
    print("\n=== Running Mini Simulation ===")
    
    M, m = 1.0, 0.001
    R = 0.2
    box_size = 10.0
    dt = 0.01
    n_steps = 100
    
    # Initialize
    M_pos = np.array([5.0, 5.0])
    M_vel = np.array([2.0, 0.0])
    m_pos = np.array([[6.0, 5.0], [7.0, 5.0]])
    m_vel = np.array([[0.0, 0.0], [0.0, 0.0]])
    
    E_initial = calculate_total_energy(M_vel, m_vel, M, m)
    collisions_detected = 0
    
    for step in range(n_steps):
        # Check collisions
        colliding = check_sphere_particle_collisions(M_pos, m_pos, R)
        if np.any(colliding):
            collisions_detected += 1
            M_vel, m_vel = handle_collision_response(
                M_pos, M_vel, m_pos, m_vel, M, m, R, colliding
            )
        
        # Update positions
        M_pos += M_vel * dt
        m_pos += m_vel * dt
        
        # Wall bounces
        M_vel = apply_wall_bounces_large_particle(M_pos, M_vel, box_size, R)
        m_vel = apply_wall_bounces_small_particles(m_pos, m_vel, box_size)
    
    E_final = calculate_total_energy(M_vel, m_vel, M, m)
    energy_drift = abs(E_final - E_initial) / E_initial
    
    print(f"Steps: {n_steps}")
    print(f"Collisions detected: {collisions_detected}")
    print(f"Initial energy: {E_initial:.6f}")
    print(f"Final energy: {E_final:.6f}")
    print(f"Energy drift: {energy_drift*100:.4f}%")
    print(f"Energy drift acceptable: {'✓' if energy_drift < 0.01 else '✗'}")
    
    return energy_drift < 0.01

if __name__ == "__main__":
    print("=" * 50)
    print("PHYSICS MODULE INTEGRATION TEST")
    print("=" * 50)
    
    tests = [
        ("Collision Conservation", test_collision_conservation),
        ("Wall Bounces", test_wall_bounces),
        ("No Collision Case", test_no_collision_case),
        ("Mini Simulation", run_mini_simulation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("Your physics module is ready to integrate!")
    else:
        print("✗ SOME TESTS FAILED")
        print("Check the output above for details")
    print("=" * 50)