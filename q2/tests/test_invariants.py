import numpy as np
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
from lu25 import simulate_markets

def test_shares_valid():
    df = simulate_markets(5, 10, "DGP1", seed=0)
    assert np.all(df["share"] > 0)
    assert np.all(df["share"] < 1)

def test_outside_share_valid():
    df = simulate_markets(5, 10, "DGP1", seed=0)
    assert np.all(df["outside_share"] >= 0)
    assert np.all(df["outside_share"] <= 1)

def test_share_sum_leq_one():
    df = simulate_markets(5, 10, "DGP1", seed=0)
    grouped = df.groupby("market_id")["share"].sum()
    assert np.all(grouped <= 1.0 + 1e-10)
