from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from .data_pipeline import FEATURE_COLUMNS, ROOT, load_dataset

EDA_DIR = ROOT / "artifacts" / "eda"


def run_eda() -> dict[str, str]:
    """Create reusable EDA plots and a data-quality summary."""
    df = load_dataset()
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    histogram_path = EDA_DIR / "feature_distributions.png"
    axes = df[FEATURE_COLUMNS].hist(figsize=(15, 11), bins=20, color="#2a788e")
    figure = axes[0][0].get_figure()
    figure.suptitle("Heart Disease Feature Distributions", fontsize=16)
    figure.tight_layout()
    figure.savefig(histogram_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    heatmap_path = EDA_DIR / "correlation_heatmap.png"
    figure, axis = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        df[[*FEATURE_COLUMNS, "target"]].corr(numeric_only=True),
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        ax=axis,
    )
    axis.set_title("Feature and Target Correlations")
    figure.tight_layout()
    figure.savefig(heatmap_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    balance_path = EDA_DIR / "class_balance.png"
    figure, axis = plt.subplots(figsize=(6, 4))
    counts = df["target"].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#d1495b", ax=axis)
    axis.set(title="Heart Disease Target Class Balance", xlabel="Target", ylabel="Records")
    for bar, count in zip(axis.patches, counts.values):
        axis.annotate(str(count), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(balance_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "rows": int(len(df)),
        "feature_count": len(FEATURE_COLUMNS),
        "missing_values": {key: int(value) for key, value in df.isna().sum().items()},
        "target_counts": {str(key): int(value) for key, value in counts.items()},
        "plots": [histogram_path.name, heatmap_path.name, balance_path.name],
    }
    summary_path = EDA_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": str(summary_path), "histograms": str(histogram_path),
            "correlations": str(heatmap_path), "class_balance": str(balance_path)}


if __name__ == "__main__":
    print(json.dumps(run_eda(), indent=2))