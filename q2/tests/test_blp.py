import numpy as np
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
from lu25 import _contraction_mapping, _simulate_choice_probs

def test_contraction_converges():
    T, J = 5, 3
    share = np.full((T, J), 0.1)
    price = np.ones((T, J))
    draws = np.random.normal(size=50)
    delta = _contraction_mapping(share, price, sigma=1.0, draws=draws)
    assert np.all(np.isfinite(delta))


def test_sigma_zero_matches_logit_formula():
    T, J = 4, 3
    delta = np.random.normal(size=(T, J))
    price = np.ones((T, J))
    draws = np.zeros(5)   # nu irrelevant when sigma=0

    s_hat = _simulate_choice_probs(delta, price, None, 0.0, 0.0, draws)

    expV = np.exp(delta - delta.max(axis=1, keepdims=True))
    denom = 1.0 + expV.sum(axis=1, keepdims=True)
    s_logit = expV / denom

    assert np.allclose(s_hat, s_logit, atol=1e-10)


