
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
PLOT_DIR = RESULTS_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load random deletion results
# --------------------------------------------------

datasets = {
    "Adult": pd.read_csv(
        RESULTS_DIR / "adult_deletion_rates.csv"
    ).rename(columns={
        "newton_runtime": "newton_time",
        "retraining_runtime": "retrain_time",
    }),

    "COMPAS": pd.read_csv(
        RESULTS_DIR / "compas_deletion_rates.csv"
    ),

    "HSLS": pd.read_csv(
        RESULTS_DIR / "hsls_deletion_rates.csv"
    ).rename(columns={
        "newton_runtime": "newton_time",
        "retraining_runtime": "retrain_time",
    }),
}


# --------------------------------------------------
# 2. Random deletion plots
# --------------------------------------------------

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


plot_metric(
    "accuracy",
    "Accuracy",
    "accuracy_vs_deletion.png",
    "Accuracy vs. Random Deletion Rate",
)

plot_metric(
    "aeod",
    "Average Equalized Odds Difference (AEOD)",
    "aeod_vs_deletion.png",
    "Fairness vs. Random Deletion Rate",
)


# Runtime comparison
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


# Parameter distance
plt.figure(figsize=(8, 5))

for dataset_name, df in datasets.items():
    plt.plot(
        df["deletion_rate"] * 100,
        df["parameter_distance"],
        marker="o",
        label=dataset_name,
    )

plt.xlabel("Deletion rate (%)")
plt.ylabel("L2 Parameter Distance (log scale)")
plt.title("Newton vs. Retrained Model Parameter Distance")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(PLOT_DIR / "parameter_distance.png", dpi=300)
plt.close()


# --------------------------------------------------
# 3. Load group deletion results
# --------------------------------------------------

group_datasets = {
    "Adult": pd.read_csv(
        RESULTS_DIR / "adult_group_deletion.csv"
    ).rename(columns={
        "newton_runtime": "newton_time",
        "retraining_runtime": "retrain_time",
        "deleted_group": "group",
    }),

    "COMPAS": pd.read_csv(
        RESULTS_DIR / "compas_group_deletion.csv"
    ).rename(columns={
        "group": "group",
    }),

    "HSLS": pd.read_csv(
        RESULTS_DIR / "hsls_group_deletion.csv"
    ).rename(columns={
        "newton_runtime": "newton_time",
        "retraining_runtime": "retrain_time",
        "deleted_group": "group",
    }),
}


# --------------------------------------------------
# 4. Group deletion plots
# --------------------------------------------------

def plot_group_metric(dataset_name, df, metric, ylabel, filename):
    plt.figure(figsize=(8, 5))

    for group_name in df["group"].unique():
        group_df = df[df["group"] == group_name].sort_values(
            "deletion_rate"
        )

        plt.plot(
            group_df["deletion_rate"] * 100,
            group_df[f"newton_{metric}"],
            marker="o",
            label=f"{group_name} - Newton",
        )

        plt.plot(
            group_df["deletion_rate"] * 100,
            group_df[f"retrained_{metric}"],
            marker="x",
            linestyle="--",
            label=f"{group_name} - Retrained",
        )

    plt.xlabel("Deletion rate within group (%)")
    plt.ylabel(ylabel)
    plt.title(f"{dataset_name}: {ylabel} vs. Group Deletion Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()


for dataset_name, df in group_datasets.items():
    plot_group_metric(
        dataset_name,
        df,
        "accuracy",
        "Accuracy",
        f"{dataset_name.lower()}_group_accuracy.png",
    )

    plot_group_metric(
        dataset_name,
        df,
        "aeod",
        "Average Equalized Odds Difference (AEOD)",
        f"{dataset_name.lower()}_group_aeod.png",
    )


# --------------------------------------------------
# 5. Completion
# --------------------------------------------------

print("All random and group deletion plots generated!")
print(f"Datasets: {', '.join(datasets.keys())}")
print(f"Saved plots to: {PLOT_DIR}")
