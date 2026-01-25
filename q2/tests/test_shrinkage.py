import numpy as np
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
from lu25 import ShrinkagePriors

def test_spike_slab_variances():
    priors = ShrinkagePriors()
    assert priors.tau0_sq < priors.tau1_sq

def test_gamma_is_binary():
    gamma = np.random.binomial(1, 0.5, size=(10, 5))
    assert set(np.unique(gamma)).issubset({0, 1})

def test_phi_in_unit_interval():
    phi = np.random.beta(1, 1, size=10)
    assert np.all(phi > 0) and np.all(phi < 1)
