
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
PLOT_DIR = RESULTS_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

adult = pd.read_csv(RESULTS_DIR / "adult_group_deletion.csv")
compas = pd.read_csv(RESULTS_DIR / "compas_group_deletion.csv")


def plot_group_metric(
    df,
    group_column,
    metric,
    ylabel,
    title,
    filename,
    original_column=None,
):
    plt.figure(figsize=(9, 5))

    for group_name, group_df in df.groupby(group_column):
        group_df = group_df.sort_values("deletion_rate")

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

    if original_column:
        plt.axhline(
            df[original_column].iloc[0],
            linestyle=":",
            label="Original baseline",
        )

    plt.xlabel("Deletion rate within demographic group (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()


# Adult: fairness after female/male deletion
plot_group_metric(
    adult,
    "deleted_group",
    "aeod",
    "AEOD",
    "Adult: Fairness After Demographic-Targeted Deletion",
    "adult_group_aeod.png",
)

# Adult: accuracy after female/male deletion
plot_group_metric(
    adult,
    "deleted_group",
    "accuracy",
    "Accuracy",
    "Adult: Accuracy After Demographic-Targeted Deletion",
    "adult_group_accuracy.png",
)

# COMPAS: fairness after race-targeted deletion
plot_group_metric(
    compas,
    "group",
    "aeod",
    "AEOD",
    "COMPAS: Fairness After Race-Targeted Deletion",
    "compas_group_aeod.png",
    original_column="original_aeod",
)

# COMPAS: accuracy after race-targeted deletion
plot_group_metric(
    compas,
    "group",
    "accuracy",
    "Accuracy",
    "COMPAS: Accuracy After Race-Targeted Deletion",
    "compas_group_accuracy.png",
    original_column="original_accuracy",
)

print("Demographic deletion plots generated successfully!")
print(f"Saved plots to: {PLOT_DIR}")
