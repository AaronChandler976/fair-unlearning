
import torch

from src.data.adult import load_adult
from src.objectives.train import train_model


def calculate_metrics(theta, X, y, sensitive):
    with torch.no_grad():
        probabilities = torch.sigmoid(X @ theta)
        predictions = (probabilities >= 0.5).long()
        labels = y.long()

        accuracy = (predictions == labels).float().mean().item()

        group_metrics = {}

        for group in (0, 1):
            mask = sensitive == group

            group_predictions = predictions[mask]
            group_labels = labels[mask]

            true_positive = (
                (group_predictions == 1) & (group_labels == 1)
            ).sum().item()

            false_negative = (
                (group_predictions == 0) & (group_labels == 1)
            ).sum().item()

            false_positive = (
                (group_predictions == 1) & (group_labels == 0)
            ).sum().item()

            true_negative = (
                (group_predictions == 0) & (group_labels == 0)
            ).sum().item()

            tpr = true_positive / max(true_positive + false_negative, 1)
            fpr = false_positive / max(false_positive + true_negative, 1)

            group_metrics[group] = {
                "TPR": tpr,
                "FPR": fpr,
            }

        # Average Equalized Odds Difference.
        aeod = 0.5 * (
            abs(group_metrics[0]["TPR"] - group_metrics[1]["TPR"])
            + abs(group_metrics[0]["FPR"] - group_metrics[1]["FPR"])
        )

    return accuracy, aeod, group_metrics


def main():
    print("Loading Adult dataset...")
    data = load_adult()

    print("\nTraining fairness-aware logistic regression...")

    theta = train_model(
        data["X_train"],
        data["y_train"],
        data["sensitive_train"],
        lambda_reg=1e-4,
        gamma=1.0,
        max_iter=500,
    )

    print("\nTraining completed!")
    print("Model parameter shape:", theta.shape)

    accuracy, aeod, group_metrics = calculate_metrics(
        theta,
        data["X_test"],
        data["y_test"],
        data["sensitive_test"],
    )

    print("\nADULT TEST RESULTS")
    print("------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AEOD: {aeod:.4f}")

    print("\nGroup fairness metrics:")
    print("Female (group 0):", group_metrics[0])
    print("Male (group 1):", group_metrics[1])

    torch.save(theta, "adult_model_original.pth")

    print("\nSaved model to adult_model_original.pth")


if __name__ == "__main__":
    main()
