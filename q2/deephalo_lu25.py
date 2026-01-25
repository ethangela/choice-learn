import sys, os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle

# -----------------------
# Import DeepHalo
# -----------------------
project_root = os.path.abspath(os.getcwd())
sys.path.append(project_root)
from deephalo import MainNetwork, log_likelihood_tf

# -----------------------
# DGP: Sparse eta + DeepHalo teacher
# -----------------------
def sample_sparse_eta(J, sparsity=0.1, scale=1.0, seed=42):
    rng = np.random.default_rng(seed)
    eta = np.zeros(J, dtype=np.float32)
    mask = rng.uniform(size=J) < sparsity
    eta[mask] = rng.normal(scale=scale, size=mask.sum())
    return eta, mask.astype(int)

def sample_choice_sets(B, J, min_size=5, seed=42):
    rng = np.random.default_rng(seed)
    E = np.zeros((B, J), dtype=np.float32)
    for b in range(B):
        k = rng.integers(min_size, J + 1)
        idx = rng.choice(J, size=k, replace=False)
        E[b, idx] = 1.0
    return E

def simulate_data(B=50000, J=20, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    E = sample_choice_sets(B, J, seed=seed)
    eta_true, eta_mask = sample_sparse_eta(J, seed=seed)

    teacher = MainNetwork(
        opt_size=J,
        depth=4,
        resnet_width=64,
        block_types=["qua","qua","qua"]
    )
    _ = teacher(tf.zeros((1, J)))

    _, logits = teacher(tf.constant(E))
    U = logits.numpy() + eta_true[None, :]
    probs = tf.nn.softmax(U, axis=1).numpy()

    y = np.zeros_like(probs)
    for b in range(B):
        y[b, np.random.choice(J, p=probs[b])] = 1.0

    return E, y, eta_true, eta_mask

# -----------------------
# DeepHalo + sparse eta (product-level)
# -----------------------
class DeepHaloWithEta(tf.keras.Model):
    def __init__(self, base_model, lambda_eta):
        super().__init__()
        self.base = base_model
        self.lambda_eta = lambda_eta
        self.eta = tf.Variable(
            tf.zeros(base_model.opt_size),
            trainable=True,
            name="eta"
        )

    def call(self, e, training=False):
        _, masked_logits = self.base(e, training=training)
        logits = masked_logits + self.eta[None, :]
        probs = tf.nn.softmax(logits, axis=-1)
        return probs, logits

    def penalty(self):
        return self.lambda_eta * tf.reduce_sum(tf.abs(self.eta))

# -----------------------
# Training utilities
# -----------------------
def train_baseline(E, y, epochs=500, lr=1e-3):
    model = MainNetwork(
        opt_size=E.shape[1],
        depth=4,
        resnet_width=64,
        block_types=["qua","qua","qua"]
    )
    opt = tf.keras.optimizers.Adam(lr)

    for _ in tqdm(range(epochs)):
        with tf.GradientTape() as tape:
            probs, _ = model(E, training=True)
            loss = log_likelihood_tf(probs, y)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
    return model

def sweep_lambda(E_tr, y_tr, E_va, y_va, lambdas, epochs=500, lr=1e-3):
    base = MainNetwork(
        opt_size=E_tr.shape[1],
        depth=4,
        resnet_width=64,
        block_types=["qua","qua","qua"]
    )
    model = DeepHaloWithEta(base, lambdas[0])
    opt = tf.keras.optimizers.Adam(lr)

    results = []
    for lam in lambdas:
        model.lambda_eta = lam
        for _ in tqdm(range(epochs)):
            with tf.GradientTape() as tape:
                probs, _ = model(E_tr, training=True)
                loss = log_likelihood_tf(probs, y_tr) + model.penalty()
            vars_ = base.trainable_variables + [model.eta]
            grads = tape.gradient(loss, vars_)
            opt.apply_gradients(zip(grads, vars_))

        probs_va, _ = model(E_va, training=False)
        nll_va = log_likelihood_tf(probs_va, y_va).numpy()
        results.append((lam, nll_va, model.eta.numpy().copy()))
    return results

# -----------------------
# Main experiment
# -----------------------
if __name__ == "__main__":
    E, y, eta_true, eta_mask = simulate_data(seed=42)
    N = E.shape[0]
    perm = np.random.permutation(N)
    tr, va = perm[:int(0.8*N)], perm[int(0.8*N):]

    E_tr, y_tr = tf.constant(E[tr]), tf.constant(y[tr])
    E_va, y_va = tf.constant(E[va]), tf.constant(y[va])

    # Baseline
    base_model = train_baseline(E_tr, y_tr)
    probs_base, _ = base_model(E_va)
    nll_base = log_likelihood_tf(probs_base, y_va).numpy()

    # Lambda sweep
    lambdas = np.logspace(-4, 0, 9)
    results = sweep_lambda(E_tr, y_tr, E_va, y_va, lambdas)

    # Extract
    nlls = [r[1] for r in results]
    etas = [r[2] for r in results]

    aucs = [
        roc_auc_score(eta_mask, np.abs(eta_hat))
        for eta_hat in etas
    ]

    # -----------------------
    # Plots
    # -----------------------
    plt.figure(figsize=(6,4))
    plt.semilogx(lambdas, nlls, marker="o", label="DeepHalo + sparse η")
    plt.axhline(nll_base, linestyle="--", color="black", label="Baseline")
    plt.xlabel(r"Sparsity penalty $\lambda_\eta$")
    plt.ylabel("Validation NLL")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nll_vs_lambda.png")

    plt.figure(figsize=(6,4))
    plt.semilogx(lambdas, aucs, marker="o")
    plt.xlabel(r"Sparsity penalty $\lambda_\eta$")
    plt.ylabel("ROC AUC (η recovery)")
    plt.tight_layout()
    plt.savefig("auc_vs_lambda.png")

    print("Baseline NLL:", nll_base)
    print("Best NLL:", min(nlls))
    print("Best AUC:", max(aucs))

    # auc_seed = []
    # for seed_num in range(42,42+7,1):
    #     print(f'current seed {seed_num}')
    #     E, y, eta_true, eta_mask = simulate_data(seed=seed_num)
    #     N = E.shape[0]
    #     perm = np.random.permutation(N)
    #     tr, va = perm[:int(0.8*N)], perm[int(0.8*N):]

    #     E_tr, y_tr = tf.constant(E[tr]), tf.constant(y[tr])
    #     E_va, y_va = tf.constant(E[va]), tf.constant(y[va])

    #     # Baseline
    #     base_model = train_baseline(E_tr, y_tr)
    #     probs_base, _ = base_model(E_va)
    #     nll_base = log_likelihood_tf(probs_base, y_va).numpy()

    #     # Lambda sweep
    #     lambdas = np.logspace(-4, 0, 9)
    #     results = sweep_lambda(E_tr, y_tr, E_va, y_va, lambdas)

    #     # Extract
    #     nlls = [r[1] for r in results]
    #     etas = [r[2] for r in results]

    #     aucs = [
    #         roc_auc_score(eta_mask, np.abs(eta_hat))
    #         for eta_hat in etas
    #     ]

    #     auc_seed.append(aucs)
    #     file_path = 'my_list_data.pkl'
    #     with open(file_path, 'wb') as file:
    #         pickle.dump(auc_seed, file)


    # # -----------------------
    # # Plots
    # # -----------------------
    # aucs = np.mean(np.array(auc_seed),axis=0)

    # plt.figure(figsize=(6,4))
    # plt.semilogx(lambdas, aucs, marker="o")
    # plt.xlabel(r"Sparsity penalty $\lambda_\eta$")
    # plt.ylabel("ROC AUC (η recovery)")
    # plt.tight_layout()
    # plt.savefig("auc_vs_lambda.png")



