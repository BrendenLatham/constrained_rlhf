import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import tqdm as tqdm
import math

def generate_dataset(states, actions, dim, samples, alpha, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    phi = torch.randn(states, actions, dim, device=device)
    phi = phi / torch.norm(phi, dim=2, keepdim=True)

    theta1_star = torch.randn(dim, device=device)
    theta1_star = theta1_star / torch.norm(theta1_star)

    theta2_star = torch.randn(dim, device=device)
    theta2_star = theta2_star / torch.norm(theta2_star)

    theta0 = alpha * theta1_star + (1 - alpha) * theta2_star

    def pi_0(x):
        logits = (phi[x] @ theta0) / eta
        probs = torch.softmax(logits, dim=0)
        return probs

    def oracle_preference(x, a1, a2, theta):
        r1 = torch.dot(phi[x, a1], theta)
        r2 = torch.dot(phi[x, a2], theta)
        p = torch.sigmoid(r1 - r2).item()
        return np.random.binomial(1, p)

    data = []
    for _ in range(samples):
        x = np.random.randint(states)
        probs = pi_0(x).cpu().numpy()
        a1, a2 = np.random.choice(actions, size=2, replace=True, p=probs)
        y1 = oracle_preference(x, a1, a2, theta1_star)
        y2 = oracle_preference(x, a1, a2, theta2_star)
        data.append((x, a1, a2, y1, y2))

    df = pd.DataFrame(data, columns=["x", "a1", "a2", "y1", "y2"])
    return df, phi, theta1_star, theta2_star, theta0

def mle_estimate(df, phi, oracle_col, dim, device):
    X, y = [], []
    for _, row in df.iterrows():
        x, a1, a2 = int(row["x"]), int(row["a1"]), int(row["a2"])
        label = int(row[oracle_col])
        phi_diff = phi[x, a1] - phi[x, a2]  # (d,)
        y_bin = 2 * label - 1
        X.append(phi_diff)
        y.append(y_bin)

    X = torch.stack(X).to(device)  # (N, d)
    y = torch.tensor(y, dtype=torch.float32, device=device)  # (N,)

    theta = torch.zeros(dim, device=device, requires_grad=True)
    optimizer = optim.LBFGS([theta], max_iter=100)

    def closure():
        optimizer.zero_grad()
        margins = y * (X @ theta)  # (N,)
        loss = torch.mean(torch.nn.functional.softplus(-margins))
        loss.backward()
        return loss

    optimizer.step(closure)
    return theta.detach()

def pi_lambda(phi, x, theta_mix, eta):
    logits = phi[x] @ theta_mix
    logits = logits - logits.max()
    probs = torch.softmax(logits / eta, dim=0)
    return probs

def evaluate_policy(df, pi, phi, theta1_star, theta2_star, theta0, eta, epsilon):
    num_states, num_actions, _ = phi.shape
    d0 = df["x"].value_counts(normalize=True).sort_index().values
    d0 = torch.tensor(d0, device=pi.device)

    J = 0.0
    KL = 0.0
    constraint_val = 0.0
    for x in range(num_states):
        pi_x = pi[x]
        logits0 = phi[x] @ theta0 / eta
        pi0_x = torch.softmax(logits0, dim=0)

        for a in range(num_actions):
            p = pi_x[a]
            log_ratio = torch.log(p + 1e-12) - torch.log(pi0_x[a] + 1e-12)
            J += d0[x] * p * torch.dot(theta1_star, phi[x, a])
            KL += d0[x] * p * log_ratio
            constraint_val += d0[x] * p * torch.dot(theta2_star, phi[x, a])

    reg_J = J - eta * KL
    constraint_violation = epsilon - constraint_val
    return reg_J.item(), constraint_violation.item()

def projected_gradient_descent(df, phi, theta1_hat, theta2_hat, eta, epsilon, Lambda, T, device):
    lam = torch.tensor(0.0, device=device)
    lambda_vals = []
    num_states, num_actions, _ = phi.shape

    counts = torch.bincount(torch.tensor(df["x"].values), minlength=num_states)
    d0 = counts.float() / counts.sum()
    d0 = d0.to(device)

    def grad_dual(lam):
        theta_mix = theta1_hat + lam * theta2_hat
        grad = 0.0
        for x in range(num_states):
            pi_x = pi_lambda(phi, x, theta_mix, eta)
            for a in range(num_actions):
                grad += d0[x] * pi_x[a] * torch.dot(theta2_hat, phi[x, a])
        return grad - epsilon

    step_size = eta / (torch.max(torch.abs(phi @ theta2_hat)) ** 2 + 1e-8)
    for _ in range(T):
        g = grad_dual(lam)
        lam = lam - step_size * g
        lam = torch.clamp(lam, 0, Lambda)
        lambda_vals.append(lam.item())

    return lambda_vals, lam.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--actions", type=int, default=10)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--T", type=int, default=100)
    args = parser.parse_args()

    global eta
    eta = args.eta

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df, phi, theta1_star, theta2_star, theta0 = generate_dataset(
        args.states, args.actions, args.dim, args.samples, args.alpha, args.seed, device
    )

    theta1_hat = mle_estimate(df, phi, "y1", args.dim, device)
    theta2_hat = mle_estimate(df, phi, "y2", args.dim, device)

    Lambda = args.lambda_init
    lambda_vals, avg_lambda = projected_gradient_descent(
        df, phi, theta1_hat, theta2_hat, args.eta, args.epsilon, Lambda, args.T, device
    )

    pi_hat = torch.stack([
        pi_lambda(phi, x, theta1_hat + avg_lambda * theta2_hat, args.eta)
        for x in range(args.states)
    ])

    J, v = evaluate_policy(df, pi_hat, phi, theta1_star, theta2_star, theta0, args.eta, args.epsilon)
    print("Learned Regularized J:", J)
    print("Constraint Violation:", v)

if __name__ == "__main__":
    main()
