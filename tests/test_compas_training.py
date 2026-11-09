
import torch

from src.data.compas import load_compas
from src.objectives.train import train_model
from tests.test_adult_unlearning import calculate_metrics


def main():
    print("Loading COMPAS dataset...")
    data = load_compas()

    print("\nTraining original COMPAS fair logistic regression model...")

    theta = train_model(
        data["X_train"],
        data["y_train"],
        data["sensitive_train"],
        lambda_reg=1e-4,
        gamma=1.0,
        max_iter=500,
    )

    accuracy, aeod = calculate_metrics(
        theta,
        data["X_test"],
        data["y_test"],
        data["sensitive_test"],
    )

    print("\n========== COMPAS BASELINE RESULTS ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AEOD:     {aeod:.4f}")

    torch.save(theta, "compas_model_original.pth")

    print("\nSaved model to compas_model_original.pth")
    print("=============================================")


if __name__ == "__main__":
    main()
