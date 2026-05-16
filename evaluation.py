import os
import pickle
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset.csv"   # apne path ke hisaab se badlo
MODEL_PATH   = "model.pkl"     # apne path ke hisaab se badlo
TEST_SIZE    = 0.2             # 20% data test ke liye
RANDOM_STATE = 42
OUTPUT_DIR   = "evaluation_results"   # plots yahan save honge
# ───────────────────────────────────────────────────────────────────────────────


def load_data(dataset_path: str):
    """Dataset load karo aur X, y alag karo."""
    print(" Dataset load ho raha hai...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset nahi mila: {dataset_path}")

    data = pd.read_csv(dataset_path, header=None, on_bad_lines="skip")
    X = data.iloc[:, :-1].values.astype(float)
    y = data.iloc[:, -1].values.astype(str)

    print(f" Total samples : {len(y)}")
    print(f" Features      : {X.shape[1]}")
    print(f" Classes       : {sorted(set(y))}\n")
    return X, y


def load_model(model_path: str):
    """Trained model load karo."""
    print(" Model load ho raha hai...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model nahi mila: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Model type: {type(model).__name__}\n")
    return model


def evaluate(model, X_test, y_test, classes):
    """Accuracy, Precision aur Classification Report nikalo."""
    print("Evaluation chal rahi hai...")
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred,
        average="weighted",   # weighted for imbalanced classes
        zero_division=0,
    )

    print("=" * 55)
    print(f"  Overall Accuracy  : {accuracy  * 100:.2f}%")
    print(f" Weighted Precision: {precision * 100:.2f}%")
    print("=" * 55)
    print()

    # Per-class report
    print("Classification Report (per class):")
    print("-" * 55)
    print(
        classification_report(
            y_test, y_pred,
            target_names=classes,
            zero_division=0,
        )
    )

    return y_pred, accuracy, precision


def plot_confusion_matrix(y_test, y_pred, classes, output_dir: str):
    """Confusion Matrix banao aur save karo."""
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalize

    n = len(classes)
    fig_size = max(10, n * 0.55)

    fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2 + 2, fig_size))
    fig.suptitle("ASL Recognition — Confusion Matrix", fontsize=16, fontweight="bold", y=1.01)

    # ── Raw counts ──
    sns.heatmap(
        cm, ax=axes[0],
        annot=True, fmt="d",
        xticklabels=classes, yticklabels=classes,
        cmap="Blues",
        linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.75},
    )
    axes[0].set_title("Raw Counts", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    # ── Normalized (%) ──
    sns.heatmap(
        cm_norm, ax=axes[1],
        annot=True, fmt=".2f",
        xticklabels=classes, yticklabels=classes,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.75, "format": mticker.PercentFormatter(xmax=1)},
    )
    axes[1].set_title("Normalized (Row %)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Confusion matrix saved → {path}")


def plot_per_class_metrics(y_test, y_pred, classes, output_dir: str):
    """Har class ki accuracy aur precision bar chart banao."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    precision_per = precision_score(y_test, y_pred, average=None, labels=classes, zero_division=0)
    recall_per    = recall_score   (y_test, y_pred, average=None, labels=classes, zero_division=0)
    f1_per        = f1_score       (y_test, y_pred, average=None, labels=classes, zero_division=0)

    x = np.arange(len(classes))
    width = 0.28

    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.7), 6))
    bars1 = ax.bar(x - width, precision_per, width, label="Precision", color="#3d7fff", alpha=0.85)
    bars2 = ax.bar(x,         recall_per,    width, label="Recall",    color="#00e5a0", alpha=0.85)
    bars3 = ax.bar(x + width, f1_per,        width, label="F1-Score",  color="#ff3d7f", alpha=0.85)

    ax.set_xlabel("ASL Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Precision / Recall / F1-Score", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.12)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "per_class_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-class metrics saved → {path}")


def plot_summary_bar(accuracy, precision, output_dir: str):
    """Overall accuracy aur precision ka simple bar chart."""
    labels  = ["Accuracy", "Precision\n(Weighted)"]
    values  = [accuracy, precision]
    colors  = ["#3d7fff", "#00e5a0"]

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(labels, values, color=colors, width=0.4, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val * 100:.2f}%",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
        )

    ax.set_ylim(0, 1.15)
    ax.set_title("Overall Model Performance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "overall_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Overall performance saved → {path}")


def main():
    print("\n" + "=" * 55)
    print("  ASL Recognition — Model Evaluation Script")
    print("=" * 55 + "\n")

    # Output folder banana
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Data load
    X, y = load_data(DATASET_PATH)

    # 2. Train/Test split (same as training ke liye consistent rehna)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # har class proportionally mile
    )
    print(f" Train samples : {len(y_train)}")
    print(f" Test  samples : {len(y_test)}\n")

    # 3. Model load
    model = load_model(MODEL_PATH)

    # 4. Classes (sorted for consistency)
    classes = sorted(set(y))

    # 5. Evaluate
    y_pred, accuracy, precision = evaluate(model, X_test, y_test, classes)

    # 6. Plots
    print(" Plots generate ho rahe hain...")
    plot_confusion_matrix(y_test, y_pred, classes, OUTPUT_DIR)
    plot_per_class_metrics(y_test, y_pred, classes, OUTPUT_DIR)
    plot_summary_bar(accuracy, precision, OUTPUT_DIR)

    print("\n Evaluation complete!")
    print(f"  All plots save in the folde : '{OUTPUT_DIR}/' .\n")


if __name__ == "__main__":
    main()