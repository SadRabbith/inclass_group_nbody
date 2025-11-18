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
