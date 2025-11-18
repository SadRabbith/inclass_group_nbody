import numpy as np
import pytest

from initialization import initialize_particles, run_simulation  

def test_no_overlap_with_big_particle():
    M_pos = np.array([5.0, 5.0])
    M_vel = np.array([0.0, 0.0])
    R = 0.2

    m_pos, m_vel = initialize_particles(
        M_pos, M_vel,
        n_particles=500,
        box_size=10.0,
        R=R
    )

    #distance from each small particle to the big one
    dist = np.linalg.norm(m_pos - M_pos, axis=1)

    #makes syre none are inside radius R + buffer
    assert np.all(dist > R), "Some particles were initialized overlapping the projectile."


def test_particles_initialized_in_box():
    M_pos = np.array([5, 5])
    m_pos, _ = initialize_particles(M_pos, np.zeros(2), 200, box_size=10, R=0.2)

    assert np.all(m_pos >= 0) and np.all(m_pos <= 10), "Particles initialized outside simulation box."


def test_big_particle_moves():
    time, M_pos_history, M_vel_history, _ = run_simulation(
        M=1.0, m=0.001, R=0.1, box_size=10.0, n_particles=10,
        dt=0.01, total_time=0.1,
        M_pos_init=np.array([2.0, 5.0]),
        M_vel_init=np.array([1.0, 0.0])
    )

    #projectile should have changed its position
    assert not np.allclose(M_pos_history[0], M_pos_history[-1]), \
        "Projectile did not move during simulation."
    

#checks for proper wall collisions

def test_wall_reflection_big_particle():
    #launch toward wall
    time, M_pos_history, M_vel_history, _ = run_simulation(
        M=1.0, m=0.001, R=0.2, box_size=5.0, n_particles=0,
        dt=0.01, total_time=1.0,
        M_pos_init=np.array([1.0, 2.5]),
        M_vel_init=np.array([-3.0, 0.0])
    )

    #velocity x-component flip sign
    vx = M_vel_history[:, 0]
    assert np.any(vx > 0), "Big particle never reflected off the wall."


def test_wall_reflection_small_particles():
    M_pos = np.array([2.0, 2.0])
    m_pos, m_vel = initialize_particles(
        M_pos, np.zeros(2),
        n_particles=50,
        box_size=10,
        R=0.3
    )

    #artificially force some particles outside the box
    m_pos[0] = np.array([-0.1, 5.0])  
    m_pos[1] = np.array([10.1, 5.0])  
    m_vel[0, 0] = -2  
    m_vel[1, 0] = 2   

    #run tiny simulation to invoke wall reflections
    _, _, _, _ = run_simulation(
        M=1.0, m=0.001, R=0.3, box_size=10.0,
        n_particles=2, dt=0.01, total_time=0.02,
        M_pos_init=np.array([2.0, 2.0]),
        M_vel_init=np.array([0.0, 0.0])
    )

    #velocities flip sign for impacted particles
    assert m_vel[0, 0] > 0, "Left-wall bounce not handled."
    assert m_vel[1, 0] < 0, "Right-wall bounce not handled."

#checks for energy conservation

def test_energy_almost_conserved():
    time, M_pos, M_vel, energy = run_simulation(
        M=1.0, m=0.001, R=0.1, box_size=10.0, n_particles=0,
        dt=0.001, total_time=1.0,
        M_pos_init=np.array([5.0, 5.0]),
        M_vel_init=np.array([1.0, 0.0])
    )

    change = abs(energy[-1] - energy[0]) / energy[0]
    assert change < 1e-3, f"Energy drift too large: {change*100:.4f}%"
