#!/usr/bin/env python3
"""
Monte Carlo replication scaffold for:
"Estimating Discrete Choice Demand Models with Sparse Market-Product Shocks" 

What this script provides:
  1) Data generator matching Section 4.1 Monte Carlo DGPs (DGP1-DGP4).
  2) A self-contained BLP-style random-coefficients logit GMM estimator for the simple 1-RC case used in the paper.
     - "BLP (with cost IV)" uses instruments (1, w, w^2, u, u^2).
     - "BLP (without cost IV)" uses instruments (1, w, w^2, w^3, w^4).
  3) An MCMC implementation for the Bayesian shrinkage approach using the paper's recommended hyperparameters (tau0^2, tau1^2) = (1e-3, 1). 

Dependencies:
  numpy, pandas, scipy, tensorflow, tensorflow_probability
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions
tfk = tfp.mcmc




'''1. DGP (data generator)'''
@dataclass
class DGPParams:
    beta_p: float = -1.0
    beta_w: float = 0.5
    sigma: float = 1.5
    xi_bar: float = -1.0
    # price equation
    u_sd: float = 0.7
    w_low: float = 1.0
    w_high: float = 2.0
    # consumers for share simulation
    N_consumers: int = 1000
    # simulation draws for choice probability integration (used by estimators)
    R_draws: int = 200


def _eta_dgp(J: int, T: int, dgp: Literal["DGP1", "DGP2", "DGP3", "DGP4"], rng: np.random.Generator) -> np.ndarray:
    """Return eta_{jt} (shape T x J) according to paper."""
    if dgp in ("DGP1", "DGP2"):
        eta = np.zeros((T, J), dtype=np.float64)
        k = int(np.floor(0.4 * J))
        # first 40% are nonzero: odd=+1, even=-1 (1-indexing in paper)
        for j in range(k):
            eta[:, j] = 1.0 if ((j + 1) % 2 == 1) else -1.0
        return eta
    elif dgp in ("DGP3", "DGP4"):
        # approx non-sparse
        return rng.normal(loc=0.0, scale=(1.0 / 3.0), size=(T, J))
    else:
        raise ValueError(f"Unknown dgp {dgp}")


def _alpha_from_eta(eta: np.ndarray, dgp: Literal["DGP1", "DGP2", "DGP3", "DGP4"]) -> np.ndarray:
    """alpha_{jt} in price equation p=alpha+0.3*w+u. Paper: alpha depends on eta for endogenous cases."""
    T, J = eta.shape
    alpha = np.zeros((T, J), dtype=np.float64)
    if dgp in ("DGP1", "DGP3"):
        # exogenous: alpha = 0
        return alpha
    if dgp == "DGP2":
        alpha[eta == 1.0] = 0.3
        alpha[eta == -1.0] = -0.3
        # alpha=0 otherwise
        return alpha
    if dgp == "DGP4":
        alpha[eta >= (1.0 / 3.0)] = 0.3
        alpha[eta <= -(1.0 / 3.0)] = -0.3
        return alpha
    raise ValueError(dgp)


def simulate_markets(
    J: int,
    T: int,
    dgp: Literal["DGP1", "DGP2", "DGP3", "DGP4"],
    params: DGPParams = DGPParams(),
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate aggregate market-product data consistent with Section 4.1. 
    Output columns:
      market_id, product_id, share, outside_share, q, price, w, u, xi_true, eta_true
    """
    rng = np.random.default_rng(seed)
    # exogenous characteristic
    w = rng.uniform(params.w_low, params.w_high, size=(T, J))
    # cost shock u
    u = rng.normal(0.0, params.u_sd, size=(T, J))
    # demand shocks
    eta = _eta_dgp(J, T, dgp, rng)
    xi = params.xi_bar + eta
    # price equation
    alpha = _alpha_from_eta(eta, dgp)
    price = alpha + 0.3 * w + u

    # simulate market shares with N_consumers draws of beta_p ~ N(beta_p, sigma^2)
    beta_p_i = rng.normal(loc=params.beta_p, scale=params.sigma, size=(T, params.N_consumers))  # market-specific draws
    # deterministic utility: beta_p_i * p_jt + beta_w * w_jt + xi_jt
    # Compute choice probs per consumer, then average.
    shares = np.zeros((T, J), dtype=np.float64)
    for t in range(T):
        V = beta_p_i[t][:, None] * price[t][None, :] + params.beta_w * w[t][None, :] + xi[t][None, :]
        # outside option normalized to 0
        expV = np.exp(V - V.max(axis=1, keepdims=True))  # stabilize
        denom = 1.0 + expV.sum(axis=1, keepdims=True)
        p_ijt = expV / denom
        shares[t] = p_ijt.mean(axis=0)
    outside_share = 1.0 - shares.sum(axis=1)

    q = np.round(shares * params.N_consumers).astype(int)  # aggregate counts (paper uses qjt)
    rows = []
    for t in range(T):
        for j in range(J):
            rows.append(
                dict(
                    market_id=t,
                    product_id=j + 1,
                    share=shares[t, j],
                    outside_share=outside_share[t],
                    q=q[t, j],
                    price=price[t, j],
                    w=w[t, j],
                    u=u[t, j],
                    xi_true=xi[t, j],
                    eta_true=eta[t, j],
                )
            )
    return pd.DataFrame(rows)




'''2. BLP estimator'''
def _simulate_choice_probs(delta: np.ndarray, price: np.ndarray, w: np.ndarray, beta_w: float, sigma: float, draws: np.ndarray) -> np.ndarray:
    """
    Predicted shares s_hat_jt given mean utility delta and sigma using simulation draws for nu ~ N(0,1).
    Here random coefficient only on price: beta_p_i = beta_p_mean + sigma * nu.
    But during contraction, we treat delta as containing beta_p_mean*price + ... so we only need sigma*nu*price term.
    """
    # shapes: delta (T,J), price (T,J), w (T,J), draws (R,)
    T, J = delta.shape
    R = draws.shape[0]
    # Utility for each draw r: delta + sigma * nu_r * price
    # Compute shares by averaging logit probabilities across draws.
    s_hat = np.zeros((T, J), dtype=np.float64)
    for r in range(R):
        nu = draws[r]
        V = delta + sigma * nu * price
        # stabilize
        Vmax = V.max(axis=1, keepdims=True)
        expV = np.exp(V - Vmax)
        denom = 1.0 + expV.sum(axis=1, keepdims=True)
        s_hat += expV / denom
    s_hat /= R
    return s_hat


def _contraction_mapping(
    share: np.ndarray,
    price: np.ndarray,
    sigma: float,
    draws: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 2000,
) -> np.ndarray:
    """Berry contraction to solve for delta given sigma."""
    T, J = share.shape
    # initialize with simple logit inversion (no random coeff): delta0 = log(s_j) - log(s0)
    s0 = 1.0 - share.sum(axis=1)
    delta = np.log(np.clip(share, 1e-12, 1.0)) - np.log(np.clip(s0[:, None], 1e-12, 1.0))
    for it in range(max_iter):
        s_hat = _simulate_choice_probs(delta, price, None, 0.0, sigma, draws)
        upd = delta + np.log(np.clip(share, 1e-12, 1.0)) - np.log(np.clip(s_hat, 1e-12, 1.0))
        diff = np.max(np.abs(upd - delta))
        delta = upd
        if diff < tol:
            break
    return delta


def _two_stage_least_squares(y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    2SLS: beta = (X'PzX)^{-1} X'Pz y, with Pz = Z (Z'Z)^{-1} Z'.
    Returns (beta, residual).
    """
    ZTZ_inv = np.linalg.pinv(Z.T @ Z)
    Pz = Z @ ZTZ_inv @ Z.T
    XTPZX = X.T @ Pz @ X
    XTPZy = X.T @ Pz @ y
    beta = np.linalg.pinv(XTPZX) @ XTPZy
    resid = y - X @ beta
    return beta, resid


def blp_gmm_estimate(
    df: pd.DataFrame,
    instruments: Literal["with_cost_iv", "without_cost_iv"],
    R_draws: int = 200,
    sigma_bounds: Tuple[float, float] = (1e-3, 5.0),
    seed: int = 123,
) -> Dict[str, object]:
    """
    One-step BLP GMM for the paper's Monte Carlo design.
    Instruments are as defined in Section 4.1. citeturn5view0
    """
    rng = np.random.default_rng(seed)
    markets = np.sort(df["market_id"].unique())
    products = np.sort(df["product_id"].unique())
    T = markets.size
    J = products.size

    # reshape into (T,J)
    share = df.pivot(index="market_id", columns="product_id", values="share").values
    price = df.pivot(index="market_id", columns="product_id", values="price").values
    w = df.pivot(index="market_id", columns="product_id", values="w").values
    u = df.pivot(index="market_id", columns="product_id", values="u").values

    # simulation draws for nu ~ N(0,1)
    draws = rng.normal(size=R_draws)

    # Build X1 = [1, price, w] stacked over jt
    y_share = share.reshape(-1)
    # We'll compute delta for each sigma, then run 2SLS for beta.
    X1 = np.column_stack([np.ones(T * J), price.reshape(-1), w.reshape(-1)])

    if instruments == "with_cost_iv":
        Z = np.column_stack(
            [
                np.ones(T * J),
                w.reshape(-1),
                (w ** 2).reshape(-1),
                u.reshape(-1),
                (u ** 2).reshape(-1),
            ]
        )
    elif instruments == "without_cost_iv":
        Z = np.column_stack(
            [
                np.ones(T * J),
                w.reshape(-1),
                (w ** 2).reshape(-1),
                (w ** 3).reshape(-1),
                (w ** 4).reshape(-1),
            ]
        )
    else:
        raise ValueError(instruments)

    # weighting matrix (one-step): (Z'Z)^{-1}
    W = np.linalg.pinv(Z.T @ Z)

    def gmm_objective(sigma: float) -> float:
        delta = _contraction_mapping(share, price, sigma, draws)
        y = delta.reshape(-1)
        beta, resid = _two_stage_least_squares(y, X1, Z)
        # moment: Z' resid
        g = (Z.T @ resid) / (T * J)
        return float(g.T @ W @ g)

    res = minimize_scalar(gmm_objective, bounds=sigma_bounds, method="bounded")
    sigma_hat = float(res.x)
    delta_hat = _contraction_mapping(share, price, sigma_hat, draws)
    beta_hat, resid_hat = _two_stage_least_squares(delta_hat.reshape(-1), X1, Z)

    out = {
        "beta_hat": beta_hat,  # [intercept(xi_bar), beta_p_mean, beta_w]
        "sigma_hat": sigma_hat,
        "xi_hat": resid_hat.reshape(T, J),
        "delta_hat": delta_hat,
        "objective": float(res.fun),
        "converged": bool(res.success),
        "optimizer": res,
    }
    return out




'''3. Bayesian shrinkage estimator'''
@dataclass
class ShrinkagePriors:
    tau0_sq: float = 1e-3
    tau1_sq: float = 1.0
    # priors per paper (uninformative)
    beta_loc: float = 0.0
    beta_scale: float = np.sqrt(10.0) 
    xi_bar_loc: float = 0.0
    xi_bar_scale: float = np.sqrt(10.0)
    # r = log sigma prior: N(0, 0.5)
    r_scale: float = np.sqrt(0.5)
    # phi prior Beta(a,b)
    a_phi: float = 1.0
    b_phi: float = 1.0


def shrinkage_mcmc(
    df: pd.DataFrame,
    priors: ShrinkagePriors = ShrinkagePriors(),
    R_draws: int = 200,
    num_results: int = 1000,
    num_burnin: int = 500,
    step_size: float = 0.02,
    seed: int = 123,
) -> Dict[str, object]:
    """
    Blocked MCMC:
      - Continuous block: (beta_p, beta_w, xi_bar, r=log sigma, eta_{jt}) sampled via NUTS given gamma.
      - Discrete block: gamma_{jt} and phi_t updated via conjugacy given eta.

    Model matches Section 3 priors and Section 4 likelihood framework.

    Returns posterior draws and posterior means.
    """
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    markets = np.sort(df["market_id"].unique())
    products = np.sort(df["product_id"].unique())
    T = markets.size
    J = products.size

    share = df.pivot(index="market_id", columns="product_id", values="share").values.astype(np.float64)
    price = df.pivot(index="market_id", columns="product_id", values="price").values.astype(np.float64)
    w = df.pivot(index="market_id", columns="product_id", values="w").values.astype(np.float64)
    q = df.pivot(index="market_id", columns="product_id", values="q").values.astype(np.float64)

    # Pre-draw nu for integrating choice probs (R0=200 in paper for MC approximation)
    nu = rng.normal(size=R_draws).astype(np.float64)  # (R,)
    nu_tf = tf.constant(nu, dtype=tf.float64)

    price_tf = tf.constant(price, dtype=tf.float64)  # (T,J)
    w_tf = tf.constant(w, dtype=tf.float64)
    q_tf = tf.constant(q, dtype=tf.float64)

    # Initialize latent indicators
    gamma = np.zeros((T, J), dtype=np.int32)  # start at 0 (spike)
    phi = np.full((T,), 0.5, dtype=np.float64)

    # helper: compute log likelihood of aggregate counts given parameters
    def log_lik(beta_p, beta_w, xi_bar, r, eta):
        sigma = tf.exp(r)
        # mean utility (T,J)
        delta = beta_p * price_tf + beta_w * w_tf + xi_bar + eta
        # integrate over nu for choice probs
        # V_r = delta + sigma * nu_r * price
        # compute logit shares for each draw, then average
        # shapes: (R,T,J)
        V = delta[None, :, :] + (sigma * nu_tf)[:, None, None] * price_tf[None, :, :]
        Vmax = tf.reduce_max(V, axis=2, keepdims=True)
        expV = tf.exp(V - Vmax)
        denom = 1.0 + tf.reduce_sum(expV, axis=2, keepdims=True)
        s_r = expV / denom  # (R,T,J)
        s_hat = tf.reduce_mean(s_r, axis=0)  # (T,J)
        # likelihood: prod_{jt} s_hat^{qjt} * s0^{q0t}; ignoring factorial constants
        s0 = 1.0 - tf.reduce_sum(s_hat, axis=1)  # (T,)
        s0 = tf.clip_by_value(s0, 1e-12, 1.0)
        s_hat = tf.clip_by_value(s_hat, 1e-12, 1.0)
        lq = tf.reduce_sum(q_tf * tf.math.log(s_hat))
        l0 = tf.reduce_sum( (1000.0 - tf.reduce_sum(q_tf, axis=1)) * tf.math.log(s0) ) 
        ll = lq + l0
        return ll

    # priors
    beta_prior = tfd.Normal(loc=priors.beta_loc, scale=priors.beta_scale)
    xi_bar_prior = tfd.Normal(loc=priors.xi_bar_loc, scale=priors.xi_bar_scale)
    r_prior = tfd.Normal(loc=0.0, scale=priors.r_scale)

    def eta_prior_logprob(eta, gamma_tf):
        # gamma=0 -> spike variance tau0_sq, gamma=1 -> slab variance tau1_sq
        dtype = eta.dtype  # ensure everything matches eta

        tau0_sq = tf.cast(priors.tau0_sq, dtype)
        tau1_sq = tf.cast(priors.tau1_sq, dtype)

        gamma_tf = tf.cast(gamma_tf, tf.int32)  # keep gamma discrete

        var = tf.where(tf.equal(gamma_tf, 0), tau0_sq, tau1_sq)  # dtype == eta.dtype
        dist = tfd.Normal(loc=tf.zeros([], dtype=dtype), scale=tf.sqrt(var))
        return tf.reduce_sum(dist.log_prob(eta))

        # var = tf.where(tf.equal(gamma_tf, 0), priors.tau0_sq, priors.tau1_sq)
        # return tf.reduce_sum(tfd.Normal(0.0, tf.sqrt(var)).log_prob(eta)) 

    def target_log_prob_fn(beta_p, beta_w, xi_bar, r, eta, gamma_tf):
        return (
            log_lik(beta_p, beta_w, xi_bar, r, eta)
            + beta_prior.log_prob(beta_p)
            + beta_prior.log_prob(beta_w)
            + xi_bar_prior.log_prob(xi_bar)
            + r_prior.log_prob(r)
            + eta_prior_logprob(eta, gamma_tf)
        )

    # initial states (continuous)
    init = [
        tf.constant(-0.5, dtype=tf.float64),  # beta_p
        tf.constant(0.3, dtype=tf.float64),   # beta_w
        tf.constant(-0.5, dtype=tf.float64),  # xi_bar
        tf.constant(np.log(1.0), dtype=tf.float64),  # r
        tf.zeros([T, J], dtype=tf.float64),   # eta
    ]

    # kernel for continuous block
    def make_kernel(gamma_tf):
        nuts = tfk.NoUTurnSampler(
            target_log_prob_fn=lambda bp, bw, xb, r, eta: target_log_prob_fn(bp, bw, xb, r, eta, gamma_tf),
            step_size=step_size,
        )
        # Adapt step size
        kernel = tfk.DualAveragingStepSizeAdaptation(
            inner_kernel=nuts, num_adaptation_steps=int(0.8 * num_burnin), target_accept_prob=0.8
        )
        return kernel

    # Run blocked Gibbs: alternate continuous NUTS and discrete gamma/phi updates
    draws = {
        "beta_p": [],
        "beta_w": [],
        "xi_bar": [],
        "sigma": [],
        "eta": [],
        "phi": [],
        "gamma": [],
    }

    state = init
    for g in tqdm(range(num_burnin + num_results)):
        gamma_tf = tf.constant(gamma, dtype=tf.int32)
        kernel = make_kernel(gamma_tf) 

        state, kr = tfk.sample_chain(
            num_results=1,
            current_state=state,
            kernel=kernel,
            num_burnin_steps=0,
            trace_fn=lambda _, pkr: pkr,
        )
        # unpack single-step result
        beta_p_s, beta_w_s, xi_bar_s, r_s, eta_s = [s[0] for s in state]
        sigma_s = tf.exp(r_s)
        state = [beta_p_s, beta_w_s, xi_bar_s, r_s, eta_s]

        # ---- Discrete updates: gamma and phi given eta ----
        # Update phi_t | gamma_t ~ Beta(a + sum_j gamma_{jt}, b + J - sum_j gamma_{jt})
        gamma_sum = gamma.sum(axis=1)
        a_post = priors.a_phi + gamma_sum
        b_post = priors.b_phi + (J - gamma_sum)
        phi = rng.beta(a_post, b_post)

        # Update gamma_{jt} | eta_{jt}, phi_t via Bernoulli with log-odds from mixture components
        eta_np = eta_s.numpy()
        # likelihood under slab/spike
        logp1 = np.log(np.clip(phi[:, None], 1e-12, 1.0)) + (-0.5*np.log(2*np.pi*priors.tau1_sq) - 0.5*(eta_np**2)/priors.tau1_sq)
        logp0 = np.log(np.clip(1.0-phi[:, None], 1e-12, 1.0)) + (-0.5*np.log(2*np.pi*priors.tau0_sq) - 0.5*(eta_np**2)/priors.tau0_sq)
        m = np.maximum(logp0, logp1)
        p1 = np.exp(logp1 - m) / (np.exp(logp0 - m) + np.exp(logp1 - m))
        gamma = rng.binomial(1, p1).astype(np.int32)


        # save after burn-in
        if g >= num_burnin:
            draws["beta_p"].append(float(beta_p_s.numpy()))
            draws["beta_w"].append(float(beta_w_s.numpy()))
            draws["xi_bar"].append(float(xi_bar_s.numpy()))
            draws["sigma"].append(float(sigma_s.numpy()))
            draws["eta"].append(eta_np.copy())
            draws["phi"].append(phi.copy())
            draws["gamma"].append(gamma.copy())

    # posterior means
    post = {
        "beta_p_mean": float(np.mean(draws["beta_p"])),
        "beta_w_mean": float(np.mean(draws["beta_w"])),
        "xi_bar_mean": float(np.mean(draws["xi_bar"])),
        "sigma_mean": float(np.mean(draws["sigma"])),
        "eta_mean": np.mean(np.stack(draws["eta"], axis=0), axis=0),
        "phi_mean": np.mean(np.stack(draws["phi"], axis=0), axis=0),
        "gamma_mean": np.mean(np.stack(draws["gamma"], axis=0), axis=0),  # inclusion probs
        "draws": draws,
    }
    return post



'''4. Table replication wrapper'''
@dataclass
class TrueParams:
    xi_bar: float = -1.0
    beta_p: float = -1.0
    beta_w: float = 0.5
    sigma: float = 1.5


def run_monte_carlo_once(
    J: int,
    T: int,
    dgp: Literal["DGP1", "DGP2", "DGP3", "DGP4"],
    seed: int,
    method: Literal["blp_with_cost_iv", "blp_without_cost_iv", "shrinkage"],
) -> Dict[str, float]:
    
    params = DGPParams()
    df = simulate_markets(J, T, dgp, params=params, seed=seed)

    xi_true = df.pivot(index="market_id", columns="product_id", values="xi_true").values

    if method == "blp_with_cost_iv":
        est = blp_gmm_estimate(df, instruments="with_cost_iv", R_draws=params.R_draws, seed=seed+999)
        beta_hat = est["beta_hat"]
        xi_hat = est["xi_hat"]
        return dict(
            Int=float(beta_hat[0]),
            beta_p=float(beta_hat[1]),
            beta_w=float(beta_hat[2]),
            sigma=float(est["sigma_hat"]),
            xi_abs_err_mean=float(np.mean(np.abs(xi_hat - xi_true))),
        )

    if method == "blp_without_cost_iv":
        est = blp_gmm_estimate(df, instruments="without_cost_iv", R_draws=params.R_draws, seed=seed+999)
        beta_hat = est["beta_hat"]
        xi_hat = est["xi_hat"]
        return dict(
            Int=float(beta_hat[0]),
            beta_p=float(beta_hat[1]),
            beta_w=float(beta_hat[2]),
            sigma=float(est["sigma_hat"]),
            xi_abs_err_mean=float(np.mean(np.abs(xi_hat - xi_true))),
        )

    if method == "shrinkage":
        post = shrinkage_mcmc(df, R_draws=params.R_draws, seed=seed+999, num_results=600, num_burnin=400) 
        xi_hat = post["xi_bar_mean"] + post["eta_mean"]
        eta_true = df.pivot(index="market_id", columns="product_id", values="eta_true").values
        gamma_mean = post["gamma_mean"]
        prob_nonzero = float(gamma_mean[eta_true != 0].mean()) if np.any(eta_true != 0) else float("nan")
        prob_zero = float(gamma_mean[eta_true == 0].mean()) if np.any(eta_true == 0) else float("nan")
        return dict(
            Int=float(post["xi_bar_mean"]),
            beta_p=float(post["beta_p_mean"]),
            beta_w=float(post["beta_w_mean"]),
            sigma=float(post["sigma_mean"]),
            xi_abs_err_mean=float(np.mean(np.abs(xi_hat - xi_true))),
            prob_nonzero=prob_nonzero,
            prob_zero=prob_zero,
        )

    raise ValueError(method)


def summarize_replications(estimates: List[Dict[str, float]], true: TrueParams) -> Dict[str, Dict[str, float]]:
    """
    Compute bias and SD across replications for:
      Int (xi_bar), beta_p, beta_w, sigma, xi (avg abs error across jt)
    """
    arr = {k: np.array([e[k] for e in estimates], dtype=float) for k in ["Int", "beta_p", "beta_w", "sigma", "xi_abs_err_mean"]}
    out = {}
    for k, true_val in [("Int", true.xi_bar), ("beta_p", true.beta_p), ("beta_w", true.beta_w), ("sigma", true.sigma)]:
        out[k] = {"Bias": float(np.mean(arr[k] - true_val)), "SD": float(np.std(arr[k], ddof=1))}
    out["xi"] = {"Bias": float(np.mean(arr["xi_abs_err_mean"])), "SD": float(np.std(arr["xi_abs_err_mean"], ddof=1))}
    return out


def run_grid(
    J: int,
    T: int,
    dgp: Literal["DGP1", "DGP2", "DGP3", "DGP4"],
    n_rep: int = 50,
    base_seed: int = 2025,
) -> pd.DataFrame:
    """
    Run a small grid (n_rep default 10 for speed) for the three methods and return a summary dataframe.
    Increase n_rep to 50 to match the paper. 
    """
    true = TrueParams()
    methods = ["blp_with_cost_iv", "blp_without_cost_iv", "shrinkage"]
    rows = []
    for method in methods:
        ests = []
        for r in range(n_rep):
            print(f'...... MC sample {r+1}/{n_rep} ......')
            est = run_monte_carlo_once(J, T, dgp, seed=base_seed + 1000 * r, method=method)
            ests.append(est)
        summ = summarize_replications(
            [{k: e[k] for k in ["Int", "beta_p", "beta_w", "sigma", "xi_abs_err_mean"]} for e in ests],
            true=true,
        )
        row = dict(
            J=J, T=T, DGP=dgp, Method=method,
            Int_Bias=summ["Int"]["Bias"], Int_SD=summ["Int"]["SD"],
            beta_p_Bias=summ["beta_p"]["Bias"], beta_p_SD=summ["beta_p"]["SD"],
            beta_w_Bias=summ["beta_w"]["Bias"], beta_w_SD=summ["beta_w"]["SD"],
            sigma_Bias=summ["sigma"]["Bias"], sigma_SD=summ["sigma"]["SD"],
            xi_Bias=summ["xi"]["Bias"], xi_SD=summ["xi"]["SD"],
        )
        if method == "shrinkage":
            row["Prob_nonzero_mean"] = float(np.mean([e["prob_nonzero"] for e in ests]))
            row["Prob_zero_mean"] = float(np.mean([e["prob_zero"] for e in ests]))
        rows.append(row)
    return pd.DataFrame(rows)



if __name__ == "__main__":

    for DGP in ["DGP1","DGP2","DGP3","DGP4"]:
        for JJ, TT in [(5,25),(5,100),(15,25),(15,100)]:
            print(f'current setting: {DGP} J{JJ} T{TT}')
            df_out = run_grid(J=JJ, T=TT, dgp=DGP, n_rep=50)
            df_out.to_pickle(f'./table_{JJ}_{TT}_{DGP}.pkl')
            unpick = pd.read_pickle(f'./table_{JJ}_{TT}_{DGP}.pkl')  
            print(unpick)

