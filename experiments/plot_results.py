
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
PLOT_DIR = RESULTS_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
datasets = {
    "Adult": pd.read_csv(RESULTS_DIR / "adult_deletion_rates.csv").rename(
        columns={
            "newton_runtime": "newton_time",
            "retraining_runtime": "retrain_time",
        }
    ),
    "COMPAS": pd.read_csv(RESULTS_DIR / "compas_deletion_rates.csv"),
}


def plot_metric(metric, ylabel, filename, title):
    plt.figure(figsize=(8, 5))

    for dataset_name, df in datasets.items():
        plt.plot(
            df["deletion_rate"] * 100,
            df[f"newton_{metric}"],
            marker="o",
            label=f"{dataset_name} - Newton",
        )
        plt.plot(
            df["deletion_rate"] * 100,
            df[f"retrained_{metric}"],
            marker="x",
            linestyle="--",
            label=f"{dataset_name} - Retrained",
        )

    plt.xlabel("Deletion rate (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()


# 1. Accuracy
plot_metric(
    "accuracy",
    "Accuracy",
    "accuracy_vs_deletion.png",
    "Accuracy vs. Random Deletion Rate",
)

# 2. Fairness
plot_metric(
    "aeod",
    "Average Equalized Odds Difference (AEOD)",
    "aeod_vs_deletion.png",
    "Fairness vs. Random Deletion Rate",
)


# 3. Runtime comparison
plt.figure(figsize=(8, 5))

for dataset_name, df in datasets.items():
    plt.plot(
        df["deletion_rate"] * 100,
        df["newton_time"],
        marker="o",
        label=f"{dataset_name} - Newton",
    )
    plt.plot(
        df["deletion_rate"] * 100,
        df["retrain_time"],
        marker="x",
        linestyle="--",
        label=f"{dataset_name} - Retraining",
    )

plt.xlabel("Deletion rate (%)")
plt.ylabel("Runtime (seconds)")
plt.title("Newton Unlearning vs. Retraining Runtime")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "runtime_comparison.png", dpi=300)
plt.close()


# 4. Parameter distance
plt.figure(figsize=(8, 5))

for dataset_name, df in datasets.items():
    plt.plot(
        df["deletion_rate"] * 100,
        df["parameter_distance"],
        marker="o",
        label=dataset_name,
    )

plt.xlabel("Deletion rate (%)")
plt.ylabel("L2 Parameter Distance")
plt.title("Newton vs. Retrained Model Parameter Distance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "parameter_distance.png", dpi=300)
plt.close()

print("All plots generated successfully!")
print(f"Saved plots to: {PLOT_DIR}")
