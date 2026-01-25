import numpy as np
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
from lu25 import _eta_dgp, simulate_markets, DGPParams

def test_eta_sparsity_dgp1():
    J, T = 10, 20
    eta = _eta_dgp(J, T, "DGP1", np.random.default_rng(0))
    nonzero_frac = np.mean(eta != 0)
    assert np.isclose(nonzero_frac, 0.4, atol=0.05)

def test_eta_exogeneity_dgp1():
    J, T = 5, 10
    df = simulate_markets(J, T, "DGP1", seed=0)
    corr = np.corrcoef(df["price"], df["eta_true"])[0, 1]
    assert abs(corr) < 0.05

def test_eta_endogeneity_dgp2():
    J, T = 5, 10
    df = simulate_markets(J, T, "DGP2", seed=0)
    corr = np.corrcoef(df["price"], df["eta_true"])[0, 1]
    assert abs(corr) > 0.1
