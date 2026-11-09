
import time
import csv
from pathlib import Path

import torch

from src.data.adult import load_adult
from src.objectives.train import train_model
from src.unlearning.newton import newton_unlearn


def calculate_metrics(theta, X, y, sensitive):
    with torch.no_grad():
        predictions = (torch.sigmoid(X @ theta) >= 0.5).long()
        labels = y.long()

        accuracy = (predictions == labels).double().mean().item()
        rates = {}

        for group in (0, 1):
            mask = sensitive == group
            pred = predictions[mask]
            actual = labels[mask]

            tp = ((pred == 1) & (actual == 1)).sum().item()
            fn = ((pred == 0) & (actual == 1)).sum().item()
            fp = ((pred == 1) & (actual == 0)).sum().item()
            tn = ((pred == 0) & (actual == 0)).sum().item()

            rates[group] = {
                "TPR": tp / max(tp + fn, 1),
                "FPR": fp / max(fp + tn, 1),
            }

        aeod = 0.5 * (
            abs(rates[0]["TPR"] - rates[1]["TPR"])
            + abs(rates[0]["FPR"] - rates[1]["FPR"])
        )

    return accuracy, aeod


def run_experiment(
    group_name,
    group_value,
    rate,
    X_train,
    y_train,
    s_train,
    X_test,
    y_test,
    s_test,
    theta_original,
):
    group_indices = torch.where(s_train == group_value)[0]

    num_delete = int(rate * len(group_indices))

    generator = torch.Generator().manual_seed(42)
    shuffled = group_indices[
        torch.randperm(len(group_indices), generator=generator)
    ]

    deleted_indices = shuffled[:num_delete]

    keep_mask = torch.ones(len(X_train), dtype=torch.bool)
    keep_mask[deleted_indices] = False

    X_remaining = X_train[keep_mask]
    y_remaining = y_train[keep_mask]
    s_remaining = s_train[keep_mask]

    print(f"\n{group_name} deletion: {rate * 100:.0f}%")
    print(f"Deleted: {num_delete}")
    print(f"Remaining: {len(X_remaining)}")

    # Newton unlearning
    start = time.perf_counter()

    theta_unlearned, _ = newton_unlearn(
        theta_original,
        X_remaining,
        y_remaining,
        s_remaining,
        lambda_reg=1e-4,
        gamma=1.0,
        damping=1e-4,
    )

    unlearning_time = time.perf_counter() - start

    # Retraining
    start = time.perf_counter()

    theta_retrained = train_model(
        X_remaining,
        y_remaining,
        s_remaining,
        lambda_reg=1e-4,
        gamma=1.0,
        max_iter=500,
    )

    retraining_time = time.perf_counter() - start

    # Evaluation
    newton_acc, newton_aeod = calculate_metrics(
        theta_unlearned, X_test, y_test, s_test
    )

    retrained_acc, retrained_aeod = calculate_metrics(
        theta_retrained, X_test, y_test, s_test
    )

    parameter_distance = torch.norm(
        theta_unlearned - theta_retrained
    ).item()

    result = {
        "deleted_group": group_name,
        "deletion_rate": rate,
        "deleted_records": num_delete,
        "remaining_records": len(X_remaining),
        "newton_accuracy": newton_acc,
        "retrained_accuracy": retrained_acc,
        "newton_aeod": newton_aeod,
        "retrained_aeod": retrained_aeod,
        "parameter_distance": parameter_distance,
        "newton_runtime": unlearning_time,
        "retraining_runtime": retraining_time,
        "speedup": retraining_time / unlearning_time,
    }

    print(f"Newton accuracy: {newton_acc:.4f}")
    print(f"Retrained accuracy: {retrained_acc:.4f}")
    print(f"Newton AEOD: {newton_aeod:.4f}")
    print(f"Retrained AEOD: {retrained_aeod:.4f}")
    print(f"Parameter distance: {parameter_distance:.8f}")
    print(f"Speedup: {result['speedup']:.2f}x")

    return result


def main():
    print("Loading Adult dataset...")
    data = load_adult()

    X_train = data["X_train"]
    y_train = data["y_train"]
    s_train = data["sensitive_train"]

    X_test = data["X_test"]
    y_test = data["y_test"]
    s_test = data["sensitive_test"]

    theta_original = torch.load(
        "adult_model_original.pth",
        map_location="cpu",
        weights_only=True,
    )

    print("\nTraining group distribution:")
    print("Female:", (s_train == 0).sum().item())
    print("Male:", (s_train == 1).sum().item())

    results = []

    # 0 = Female, 1 = Male
    groups = [
        ("Female", 0),
        ("Male", 1),
    ]

    rates = [0.01, 0.05, 0.10]

    for group_name, group_value in groups:
        for rate in rates:
            result = run_experiment(
                group_name,
                group_value,
                rate,
                X_train,
                y_train,
                s_train,
                X_test,
                y_test,
                s_test,
                theta_original,
            )
            results.append(result)

    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "adult_group_deletion.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nAll group deletion experiments completed.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
