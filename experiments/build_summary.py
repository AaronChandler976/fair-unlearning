
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("experiments/results")

# Baseline metrics from the original models.
BASELINES = {
    "Adult": {"accuracy": 0.8446, "aeod": 0.0602},
    "COMPAS": {"accuracy": 0.6525, "aeod": 0.1469},
    "HSLS": {"accuracy": 0.8034, "aeod": 0.0389},
}

# Standardize runtime and group column names across datasets.
def load_results(filename, group=False):
    df = pd.read_csv(RESULTS_DIR / filename)

    rename = {
        "newton_runtime": "newton_time",
        "retraining_runtime": "retrain_time",
        "deleted_group": "group",
    }

    df = df.rename(columns=rename)

    return df


random_files = {
    "Adult": "adult_deletion_rates.csv",
    "COMPAS": "compas_deletion_rates.csv",
    "HSLS": "hsls_deletion_rates.csv",
}

group_files = {
    "Adult": "adult_group_deletion.csv",
    "COMPAS": "compas_group_deletion.csv",
    "HSLS": "hsls_group_deletion.csv",
}


def summarize(dataset, strategy, filename):
    df = load_results(filename)

    baseline = BASELINES[dataset]

    newton_runtime = df["newton_time"]
    retrain_runtime = df["retrain_time"]

    speedups = retrain_runtime / newton_runtime

    return {
        "Dataset": dataset,
        "Deletion strategy": strategy,
        "Scenarios": len(df),

        "Baseline accuracy": baseline["accuracy"],
        "Baseline AEOD": baseline["aeod"],

        "Newton accuracy (mean)": df["newton_accuracy"].mean(),
        "Retrained accuracy (mean)": df["retrained_accuracy"].mean(),

        "Newton AEOD (mean)": df["newton_aeod"].mean(),
        "Retrained AEOD (mean)": df["retrained_aeod"].mean(),

        "Parameter distance (mean)": df["parameter_distance"].mean(),

        "Newton runtime (median s)": newton_runtime.median(),
        "Retraining runtime (median s)": retrain_runtime.median(),
        "Speedup (median x)": speedups.median(),
    }


# Build six summary rows: three datasets × two deletion strategies.
rows = []

for dataset, filename in random_files.items():
    rows.append(
        summarize(dataset, "Random deletion", filename)
    )

for dataset, filename in group_files.items():
    rows.append(
        summarize(dataset, "Group deletion", filename)
    )

summary = pd.DataFrame(rows)

# Save machine-readable results.
csv_path = RESULTS_DIR / "consolidated_results.csv"
summary.to_csv(csv_path, index=False, float_format="%.6f")

# Create a readable Markdown table.
display = summary.copy()

for column in display.columns:
    if column not in ["Dataset", "Deletion strategy", "Scenarios"]:
        display[column] = display[column].map(
            lambda value: f"{value:.4f}"
        )

md_path = RESULTS_DIR / "consolidated_results.md"

with open(md_path, "w", encoding="utf-8") as file:
    file.write("# Consolidated Fair Machine Unlearning Results\n\n")
    file.write(
        "Means are calculated across the tested deletion scenarios. "
        "Group deletion averages include both protected groups. "
        "Runtime statistics summarize single-run measurements and "
        "should not be interpreted as repeated benchmark estimates.\n\n"
    )

    file.write("| " + " | ".join(display.columns) + " |\n")
    file.write("| " + " | ".join(["---"] * len(display.columns)) + " |\n")

    for _, row in display.iterrows():
        file.write(
            "| " + " | ".join(str(value) for value in row) + " |\n"
        )

print("Consolidated results generated!")
print(f"CSV: {csv_path}")
print(f"Markdown: {md_path}")
print("\nSummary:")
print(display.to_string(index=False))
