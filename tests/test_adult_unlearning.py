
import time
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


def main():
    torch.manual_seed(42)

    print("Loading Adult dataset...")
    data = load_adult()

    X_train = data["X_train"]
    y_train = data["y_train"]
    s_train = data["sensitive_train"]

    X_test = data["X_test"]
    y_test = data["y_test"]
    s_test = data["sensitive_test"]

    # Load the original model trained on all records.
    theta_original = torch.load(
        "adult_model_original.pth",
        map_location="cpu",
        weights_only=True,
    )

    # Randomly delete 1% of the training records.
    num_delete = int(0.01 * len(X_train))

    generator = torch.Generator().manual_seed(42)
    deleted_indices = torch.randperm(
        len(X_train), generator=generator
    )[:num_delete]

    keep_mask = torch.ones(len(X_train), dtype=torch.bool)
    keep_mask[deleted_indices] = False

    X_remaining = X_train[keep_mask]
    y_remaining = y_train[keep_mask]
    s_remaining = s_train[keep_mask]

    print(f"\nOriginal training records: {len(X_train)}")
    print(f"Deleted records: {num_delete}")
    print(f"Remaining records: {len(X_remaining)}")

    # Experiment 1: Newton unlearning.
    print("\nRunning Newton unlearning...")

    start = time.perf_counter()

    theta_unlearned, info = newton_unlearn(
        theta_original,
        X_remaining,
        y_remaining,
        s_remaining,
        lambda_reg=1e-4,
        gamma=1.0,
        damping=1e-4,
    )

    unlearning_time = time.perf_counter() - start

    # Experiment 2: Retraining from scratch.
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

    # Experiment 3: Compare all three models.
    original_acc, original_aeod = calculate_metrics(
        theta_original, X_test, y_test, s_test
    )

    unlearned_acc, unlearned_aeod = calculate_metrics(
        theta_unlearned, X_test, y_test, s_test
    )

    retrained_acc, retrained_aeod = calculate_metrics(
        theta_retrained, X_test, y_test, s_test
    )

    parameter_distance = torch.norm(
        theta_unlearned - theta_retrained
    ).item()

    print("\n========== ADULT UNLEARNING RESULTS ==========")

    print(f"\nDeleted records: {num_delete} ({100 * num_delete / len(X_train):.2f}%)")

    print("\nAccuracy:")
    print(f"Original:        {original_acc:.4f}")
    print(f"Newton:          {unlearned_acc:.4f}")
    print(f"Retrained:       {retrained_acc:.4f}")

    print("\nAEOD:")
    print(f"Original:        {original_aeod:.4f}")
    print(f"Newton:          {unlearned_aeod:.4f}")
    print(f"Retrained:       {retrained_aeod:.4f}")

    print("\nParameter distance (Newton vs retrained):")
    print(f"{parameter_distance:.8f}")

    print("\nRuntime:")
    print(f"Newton unlearning: {unlearning_time:.4f} seconds")
    print(f"Retraining:        {retraining_time:.4f} seconds")

    print("\n==============================================")


if __name__ == "__main__":
    main()
