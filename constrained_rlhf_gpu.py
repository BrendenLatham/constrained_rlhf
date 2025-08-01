#!/usr/bin/env python3

import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------------------------------------------------- #
# 1.  DATA GENERATION                                                         #
# --------------------------------------------------------------------------- #
def generate_dataset(states, actions, dim, samples, alpha, eta, seed, device):
    """
    Returns
        df              – preference dataset
        phi             – (state, action) feature tensor         (S, A, d)
        theta1_star     – true reward parameter for oracle 1     (d,)
        theta2_star     – true reward parameter for oracle 2     (d,)
        theta0          – mixture parameter that defines pi_0.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # random unit-norm features for each (x,a)
    phi = torch.randn(states, actions, dim, device=device)
    phi = phi / torch.norm(phi, dim=2, keepdim=True)

    # ground-truth reward parameters with unit norm
    theta1_star = torch.randn(dim, device=device)
    theta1_star = theta1_star / torch.norm(theta1_star)
    theta2_star = torch.randn(dim, device=device)
    theta2_star = theta2_star / torch.norm(theta2_star)

    theta0 = alpha * theta1_star + (1.0 - alpha) * theta2_star  # behaviour param

    # ----- helper functions ------------------------------------------------ #
    def pi_0(x):
        logits = phi[x] @ theta0              # (A,)
        logits = logits - logits.max()        # stability
        return torch.softmax(logits / eta, dim=0)

    def oracle_preference(x, a1, a2, theta):
        r1 = torch.dot(phi[x, a1], theta)
        r2 = torch.dot(phi[x, a2], theta)
        p  = torch.sigmoid(r1 - r2).item()
        return np.random.binomial(1, p)       # 1 ⇔ (a1 ≻ a2)

    # ----- draw dataset ---------------------------------------------------- #
    rows = []
    for _ in range(samples):
        x     = np.random.randint(states)
        probs = pi_0(x).cpu().numpy()
        a1, a2 = np.random.choice(actions, size=2, replace=True, p=probs)
        y1 = oracle_preference(x, a1, a2, theta1_star)
        y2 = oracle_preference(x, a1, a2, theta2_star)
        rows.append((x, a1, a2, y1, y2))

    df = pd.DataFrame(rows, columns=["x", "a1", "a2", "y1", "y2"])
    return df, phi, theta1_star, theta2_star, theta0


# --------------------------------------------------------------------------- #
# 2.  MLE FOR EACH ORACLE                                                     #
# --------------------------------------------------------------------------- #
def mle_estimate(df, phi, oracle_col, dim, device):
    """Logistic-loss MLE for pairwise comparisons."""
    X_list, y_list = [], []
    for _, row in df.iterrows():
        x, a1, a2 = map(int, (row["x"], row["a1"], row["a2"]))
        label = int(row[oracle_col])
        phi_diff = phi[x, a1] - phi[x, a2]    # (d,)
        y_bin = 2 * label - 1                 # {0,1} → {-1,+1}
        X_list.append(phi_diff)
        y_list.append(y_bin)

    X = torch.stack(X_list).to(device)        # (N, d)
    y = torch.tensor(y_list, dtype=torch.float32, device=device)  # (N,)

    theta = torch.zeros(dim, device=device, requires_grad=True)
    optimizer = optim.LBFGS([theta], max_iter=100)

    def closure():
        optimizer.zero_grad()
        margins = y * (X @ theta)
        loss = torch.mean(nn.functional.softplus(-margins))
        loss.backward()
        return loss

    optimizer.step(closure)
    return theta.detach()


# --------------------------------------------------------------------------- #
# 3.  POLICIES                                                                #
# --------------------------------------------------------------------------- #
def pi_lambda(phi, x, theta_mix, eta):
    logits = phi[x] @ theta_mix
    logits = logits - logits.max()
    return torch.softmax(logits / eta, dim=0)


# --------------------------------------------------------------------------- #
# 4.  EVALUATION                                                              #
# --------------------------------------------------------------------------- #
def evaluate_policy(df, pi, phi,
                    theta1_star, theta2_star, theta0,
                    eta, epsilon):
    states, actions, _ = phi.shape

    # empirical state distribution of the dataset (pads missing states)
    counts = torch.bincount(torch.tensor(df["x"].values),
                            minlength=states).float()
    d0 = counts / counts.sum()                # (S,)

    J = KL = constraint = 0.0
    for x in range(states):
        pi_x = pi[x]                          # (A,)

        logits0 = phi[x] @ theta0
        logits0 = logits0 - logits0.max()
        pi0_x = torch.softmax(logits0 / eta, dim=0)

        for a in range(actions):
            p = pi_x[a]
            log_ratio = torch.log(p + 1e-12) - torch.log(pi0_x[a] + 1e-12)

            J += d0[x] * p * torch.dot(theta1_star, phi[x, a])
            KL += d0[x] * p * log_ratio
            constraint += d0[x] * p * torch.dot(theta2_star, phi[x, a])

    regularised_J = J - eta * KL
    constraint_violation = epsilon - constraint
    return regularised_J.item(), constraint_violation.item()


# --------------------------------------------------------------------------- #
# 5.  DUAL ASCENT / PGD                                                       #
# --------------------------------------------------------------------------- #
def projected_gradient_descent(df, phi,
                               theta1_hat, theta2_hat,
                               eta, epsilon,
                               lambda_cap, steps,
                               device,
                               early_stop_tol=None):
    """
    Vectorised PGD: O(S·A) per step instead of O(S·A + Python).
    """
    lam = torch.tensor(0.0, device=device)         # dual variable
    lambda_hist = []

    states, actions, _ = phi.shape

    counts = torch.bincount(
        torch.tensor(df["x"].values), minlength=states
    ).float()
    d0 = (counts / counts.sum()).to(device)        # (S,)

    # Pre-compute feature–theta2 term once: (S, A)
    phi_theta2 = phi @ theta2_hat

    # Safe step-size (same derivation)
    L = torch.max(torch.abs(phi_theta2))
    step_size = eta / (L ** 2 + 1e-8)

    for _ in range(steps):
        # ---- compute grad g(λ) in a single kernel ------------------------ #
        theta_mix = theta1_hat + lam * theta2_hat            # (d,)
        logits = phi @ theta_mix                             # (S, A)
        logits = logits - logits.max(dim=1, keepdim=True).values
        pi = torch.softmax(logits / eta, dim=1)              # (S, A)

        g = torch.sum(d0[:, None] * pi * phi_theta2) - epsilon   # scalar
        lam = lam - step_size * g
        lam = torch.clamp(lam, 0.0, lambda_cap)

        lambda_hist.append(lam.item())

        if early_stop_tol is not None and torch.abs(g) < early_stop_tol:
            break

    return lambda_hist, lam.item()



# --------------------------------------------------------------------------- #
# 6.  MAIN                                                                    #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states",   type=int,   default=100)
    parser.add_argument("--actions",  type=int,   default=10)
    parser.add_argument("--dim",      type=int,   default=10)
    parser.add_argument("--samples",  type=int,   default=100)
    parser.add_argument("--alpha",    type=float, default=0.5)
    parser.add_argument("--eta",      type=float, default=0.05)
    parser.add_argument("--epsilon",  type=float, default=0.5)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--device",   type=str,   default="cuda")
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--T",        type=int,   default=100)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. synthetic dataset --------------------------------------------------- #
    df, phi, theta1_star, theta2_star, theta0 = generate_dataset(
        args.states, args.actions, args.dim, args.samples,
        args.alpha, args.eta, args.seed, device
    )

    # 2. estimate reward parameters ----------------------------------------- #
    theta1_hat = mle_estimate(df, phi, "y1", args.dim, device)
    theta2_hat = mle_estimate(df, phi, "y2", args.dim, device)

    # 3. optimise dual variable λ ------------------------------------------- #
    lambda_hist, lambda_opt = projected_gradient_descent(
        df, phi, theta1_hat, theta2_hat,
        args.eta, args.epsilon,
        args.lambda_init, args.T, device
    )

    # 4. learned policy ------------------------------------------------------ #
    pi_hat = torch.stack([
        pi_lambda(phi, x, theta1_hat + lambda_opt * theta2_hat, args.eta)
        for x in range(args.states)
    ])

    J_val, violation = evaluate_policy(
        df, pi_hat, phi,
        theta1_star, theta2_star, theta0,
        args.eta, args.epsilon
    )

    print(f"Learned regularised J   : {J_val:.4f}")
    print(f"Constraint violation    : {violation:.4f}")


if __name__ == "__main__":
    main()
