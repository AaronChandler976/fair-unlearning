
import csv
import time
from pathlib import Path

import torch

from src.data.compas import load_compas
from src.objectives.train import train_model
from src.unlearning.newton import newton_unlearn
from tests.test_adult_unlearning import calculate_metrics


def main():
    print("Loading COMPAS dataset...")
    data = load_compas()

    X = data["X_train"]
    y = data["y_train"]
    sensitive = data["sensitive_train"]

    X_test = data["X_test"]
    y_test = data["y_test"]
    sensitive_test = data["sensitive_test"]

    theta_original = torch.load(
        "compas_model_original.pth", map_location="cpu"
    )

    original_accuracy, original_aeod = calculate_metrics(
        theta_original, X_test, y_test, sensitive_test
    )

    groups = {
        0: "Caucasian",
        1: "African-American",
    }

    deletion_rates = [0.01, 0.05, 0.10]
    results = []

    for group_id, group_name in groups.items():
        group_indices = torch.where(sensitive == group_id)[0]

        generator = torch.Generator().manual_seed(42)
        permutation = group_indices[
            torch.randperm(len(group_indices), generator=generator)
        ]

        print(f"\n{'=' * 55}")
        print(f"Deleting records from: {group_name}")
        print(f"Group training records: {len(group_indices)}")
        print("=" * 55)

        for rate in deletion_rates:
            num_deleted = int(len(group_indices) * rate)
            deleted_indices = permutation[:num_deleted]

            keep_mask = torch.ones(len(X), dtype=torch.bool)
            keep_mask[deleted_indices] = False

            X_remaining = X[keep_mask]
            y_remaining = y[keep_mask]
            sensitive_remaining = sensitive[keep_mask]

            print(f"\n{rate:.0%} group deletion")
            print(f"Deleted: {num_deleted}")
            print(f"Remaining: {len(X_remaining)}")

            start = time.perf_counter()

            theta_newton, _ = newton_unlearn(
                theta_original,
                X_remaining,
                y_remaining,
                sensitive_remaining,
                lambda_reg=1e-4,
                gamma=1.0,
            )

            newton_time = time.perf_counter() - start

            start = time.perf_counter()

            theta_retrained = train_model(
                X_remaining,
                y_remaining,
                sensitive_remaining,
                lambda_reg=1e-4,
                gamma=1.0,
                max_iter=500,
            )

            retrain_time = time.perf_counter() - start

            newton_accuracy, newton_aeod = calculate_metrics(
                theta_newton, X_test, y_test, sensitive_test
            )

            retrained_accuracy, retrained_aeod = calculate_metrics(
                theta_retrained, X_test, y_test, sensitive_test
            )

            parameter_distance = torch.norm(
                theta_newton - theta_retrained
            ).item()

            speedup = retrain_time / newton_time

            print(f"Newton accuracy:    {newton_accuracy:.4f}")
            print(f"Retrained accuracy: {retrained_accuracy:.4f}")
            print(f"Newton AEOD:        {newton_aeod:.4f}")
            print(f"Retrained AEOD:     {retrained_aeod:.4f}")
            print(f"Parameter distance: {parameter_distance:.8f}")
            print(f"Speedup:            {speedup:.2f}x")

            results.append({
                "group": group_name,
                "deletion_rate": rate,
                "num_deleted": num_deleted,
                "original_accuracy": original_accuracy,
                "newton_accuracy": newton_accuracy,
                "retrained_accuracy": retrained_accuracy,
                "original_aeod": original_aeod,
                "newton_aeod": newton_aeod,
                "retrained_aeod": retrained_aeod,
                "parameter_distance": parameter_distance,
                "newton_time": newton_time,
                "retrain_time": retrain_time,
                "speedup": speedup,
            })

    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "compas_group_deletion.csv"

    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()
