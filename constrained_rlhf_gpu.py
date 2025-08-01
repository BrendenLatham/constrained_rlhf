import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse


def generate_offline_rlhf_dataset(num_states, num_actions, d, N, alpha, eta, seed, device):
    torch.manual_seed(seed)
    phi = torch.randn(num_states, num_actions, d, device=device)
    phi = phi / phi.norm(dim=2, keepdim=True)

    theta1_star = torch.randn(d, device=device)
    theta1_star /= theta1_star.norm()

    theta2_star = torch.randn(d, device=device)
    theta2_star /= theta2_star.norm()

    theta0 = alpha * theta1_star + (1 - alpha) * theta2_star
    logits = torch.einsum("sad,d->sa", phi, theta0)
    pi_0_probs = F.softmax(logits / eta, dim=1)

    x = torch.randint(0, num_states, (N,), device=device)
    pi_x = pi_0_probs[x]
    a_choices = torch.multinomial(pi_x, 2, replacement=True)
    a1, a2 = a_choices[:, 0], a_choices[:, 1]

    def oracle_pref(theta):
        r1 = (phi[x, a1] * theta).sum(dim=1)
        r2 = (phi[x, a2] * theta).sum(dim=1)
        probs = torch.sigmoid(r1 - r2)
        return torch.bernoulli(probs).long()

    y1 = oracle_pref(theta1_star)
    y2 = oracle_pref(theta2_star)

    df = pd.DataFrame({
        "x": x.cpu().numpy(),
        "a1": a1.cpu().numpy(),
        "a2": a2.cpu().numpy(),
        "y1": y1.cpu().numpy(),
        "y2": y2.cpu().numpy(),
    })
    return df, phi, theta1_star, theta2_star, theta0


def mle_estimate(df, phi, oracle_col, d, reg=1e-4, device="cuda"):
    X, y = [], []
    for _, row in df.iterrows():
        x, a1, a2, label = int(row["x"]), int(row["a1"]), int(row["a2"]), int(row[oracle_col])
        phi_diff = phi[x, a1] - phi[x, a2]
        y_bin = 2 * label - 1
        X.append(phi_diff)
        y.append(y_bin)
    X = torch.stack(X).to(device)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    theta = torch.zeros(d, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([theta], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        margins = y * X @ theta
        log_likelihood = -torch.mean(F.softplus(-margins))
        reg_term = 0.5 * reg * torch.sum(theta ** 2)
        loss = -log_likelihood + reg_term
        loss.backward()
        return loss

    optimizer.step(closure)
    return theta.detach()


def compute_gibbs_policy(phi, theta_mix, eta):
    logits = torch.einsum("sad,d->sa", phi, theta_mix)
    probs = F.softmax(logits / eta, dim=1)
    return probs


def compute_regularized_objective(pi, phi, theta1_star, theta0, eta, state_dist):
    logits0 = torch.einsum("sad,d->sa", phi, theta0) / eta
    pi0 = F.softmax(logits0, dim=1)
    log_ratio = torch.log(pi + 1e-12) - torch.log(pi0 + 1e-12)

    reward_term = (pi * torch.einsum("sad,d->sa", phi, theta1_star)).sum(dim=1)
    kl_term = (pi * log_ratio).sum(dim=1)
    J = (reward_term - eta * kl_term) @ state_dist
    return J.item()


def compute_constraint_violation(pi, phi, theta2_star, epsilon, state_dist):
    r2_expectation = (pi * torch.einsum("sad,d->sa", phi, theta2_star)).sum(dim=1)
    constraint_val = r2_expectation @ state_dist
    return epsilon - constraint_val.item(), constraint_val.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--actions", type=int, default=10)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--step_size", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    df, phi, theta1_star, theta2_star, theta0 = generate_offline_rlhf_dataset(
        args.states, args.actions, args.dim, args.samples,
        args.alpha, args.eta, args.seed, device
    )

    theta1_hat = mle_estimate(df, phi, "y1", args.dim, device=device)
    theta2_hat = mle_estimate(df, phi, "y2", args.dim, device=device)

    print("\nMLE estimation complete.")

    theta_mix = theta1_hat + theta2_hat  # Example: λ=1
    pi_hat = compute_gibbs_policy(phi, theta_mix, args.eta)

    state_counts = df["x"].value_counts(normalize=True).sort_index()
    state_dist = torch.tensor([state_counts.get(i, 0.0) for i in range(args.states)], device=device)

    J_hat = compute_regularized_objective(pi_hat, phi, theta1_star, theta0, args.eta, state_dist)
    violation, constraint_val = compute_constraint_violation(pi_hat, phi, theta2_star, args.epsilon, state_dist)

    print("\nLearned policy performance:")
    print("J(pi):", J_hat)
    print("Constraint value:", constraint_val)
    print("Constraint violation:", violation)


if __name__ == "__main__":
    main()
