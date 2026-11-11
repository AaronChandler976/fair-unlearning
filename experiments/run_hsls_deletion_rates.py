
import time
import csv
from pathlib import Path

import torch

from src.data.hsls import load_hsls
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


def main():
    torch.manual_seed(42)

    print("Loading HSLS dataset...")
    data = load_hsls()

    X_train = data["X_train"]
    y_train = data["y_train"]
    s_train = data["sensitive_train"]

    X_test = data["X_test"]
    y_test = data["y_test"]
    s_test = data["sensitive_test"]

    theta_original = torch.load(
        "hsls_model_original.pth",
        map_location="cpu",
        weights_only=True,
    )

    original_acc, original_aeod = calculate_metrics(
        theta_original, X_test, y_test, s_test
    )

    # One fixed random ordering makes deletion sets nested:
    # the 1% deleted records are included in the 5% and 10% sets.
    generator = torch.Generator().manual_seed(42)
    random_order = torch.randperm(
        len(X_train), generator=generator
    )

    deletion_rates = [0.01, 0.05, 0.10]
    results = []

    for rate in deletion_rates:
        print(f"\n{'=' * 55}")
        print(f"Running experiment: {rate * 100:.0f}% deletion")
        print("=" * 55)

        num_delete = int(rate * len(X_train))
        deleted_indices = random_order[:num_delete]

        keep_mask = torch.ones(len(X_train), dtype=torch.bool)
        keep_mask[deleted_indices] = False

        X_remaining = X_train[keep_mask]
        y_remaining = y_train[keep_mask]
        s_remaining = s_train[keep_mask]

        print(f"Deleted: {num_delete}")
        print(f"Remaining: {len(X_remaining)}")

        # Newton unlearning
        print("\nRunning Newton unlearning...")
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

        # Retraining from scratch
        print("Retraining from scratch...")
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

        # Evaluate all models on the same untouched test set
        unlearned_acc, unlearned_aeod = calculate_metrics(
            theta_unlearned, X_test, y_test, s_test
        )

        retrained_acc, retrained_aeod = calculate_metrics(
            theta_retrained, X_test, y_test, s_test
        )

        parameter_distance = torch.norm(
            theta_unlearned - theta_retrained
        ).item()

        speedup = retraining_time / unlearning_time

        result = {
            "deletion_rate": rate,
            "deleted_records": num_delete,
            "remaining_records": len(X_remaining),
            "original_accuracy": original_acc,
            "newton_accuracy": unlearned_acc,
            "retrained_accuracy": retrained_acc,
            "original_aeod": original_aeod,
            "newton_aeod": unlearned_aeod,
            "retrained_aeod": retrained_aeod,
            "parameter_distance": parameter_distance,
            "newton_runtime": unlearning_time,
            "retraining_runtime": retraining_time,
            "speedup": speedup,
        }

        results.append(result)

        print(f"\nAccuracy:")
        print(f"Original:  {original_acc:.4f}")
        print(f"Newton:    {unlearned_acc:.4f}")
        print(f"Retrained: {retrained_acc:.4f}")

        print(f"\nAEOD:")
        print(f"Original:  {original_aeod:.4f}")
        print(f"Newton:    {unlearned_aeod:.4f}")
        print(f"Retrained: {retrained_aeod:.4f}")

        print(f"\nParameter distance: {parameter_distance:.8f}")
        print(f"Newton runtime:     {unlearning_time:.4f}s")
        print(f"Retraining runtime: {retraining_time:.4f}s")
        print(f"Speedup:            {speedup:.2f}x")

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "hsls_deletion_rates.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll experiments completed.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
