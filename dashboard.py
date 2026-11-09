import json
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# DeepLearningPro - LULC Dashboard
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

# If dashboard.py is copied into the project root, this works directly.
# Otherwise, allow the user to select the project directory in the sidebar.
DEFAULT_PROJECT_DIR = PROJECT_DIR

st.set_page_config(
    page_title="Sentinel-1 LULC Dashboard",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Sentinel-1 Land-Use / Land-Cover Classification")
st.caption("DeepLearningPro — SmallUNet pixel-classification project")

# ---------------------------------------------------------
# Locate project/results
# ---------------------------------------------------------

project_dir = DEFAULT_PROJECT_DIR

if not (project_dir / "results").exists():
    st.error(
        "The dashboard must be placed in the DeepLearningPro project root "
        "so that the 'results' directory is available."
    )
    st.stop()

results_root = project_dir / "results"

# Prefer the latest completed evaluation directory containing FINAL_PROJECT_RESULTS.json
evaluation_dirs = sorted(
    results_root.glob("**/test_evaluation_*/"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

evaluation_dir = None
for candidate in evaluation_dirs:
    if (candidate / "FINAL_PROJECT_RESULTS.json").exists():
        evaluation_dir = candidate
        break

if evaluation_dir is None:
    st.error("No completed FINAL_PROJECT_RESULTS.json was found under results/.")
    st.stop()

# ---------------------------------------------------------
# Load evaluation data
# ---------------------------------------------------------

summary_path = evaluation_dir / "test_summary.json"
classes_path = evaluation_dir / "test_classes.csv"
confusion_path = evaluation_dir / "test_confusion.csv"
final_results_path = evaluation_dir / "FINAL_PROJECT_RESULTS.json"

summary = {}
if summary_path.exists():
    with open(summary_path, "r") as f:
        summary = json.load(f)

classes_df = pd.read_csv(classes_path) if classes_path.exists() else pd.DataFrame()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Project")
st.sidebar.write(f"**Evaluation:** `{evaluation_dir.name}`")
st.sidebar.write(f"**Model run:** `{evaluation_dir.parent.name}`")

page = st.sidebar.radio(
    "Dashboard section",
    [
        "Overview",
        "Model Performance",
        "Per-Class Analysis",
        "Confusion Matrix",
        "Monthly Prototype",
        "Limitations",
    ],
)

# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def pct(value):
    return f"{float(value) * 100:.2f}%"

def find_pngs():
    return sorted(
        results_root.glob("**/*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------

if page == "Overview":

    st.header("Project Overview")

    accuracy = summary.get("overall_accuracy", 0)
    miou = summary.get("mean_iou", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", pct(accuracy))
    c2.metric("Mean IoU", pct(miou))
    c3.metric("Majority Baseline", "17.71%")
    c4.metric("Test Patches", "5,050")

    st.markdown("---")

    st.subheader("Project Pipeline")

    st.markdown(
        """
        **Sentinel-1 imagery → preprocessing → 256×256 pixel patches → "
        "**SmallUNet → independent test evaluation → monthly Sentinel-1 prototype**

        The project uses VV/VH Sentinel-1 imagery for pixel-level
        land-cover classification across nine LULC classes.
        """
    )

    st.subheader("Final Test Results")

    if not classes_df.empty:
        display_df = classes_df[
            ["class_id", "class_name", "iou", "f1"]
        ].copy()
        display_df["iou"] = display_df["iou"].map(lambda x: f"{x * 100:.2f}%")
        display_df["f1"] = display_df["f1"].map(lambda x: f"{x * 100:.2f}%")
        display_df.columns = ["ID", "Class", "IoU", "F1"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Model Performance
# ---------------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance")

    accuracy = summary.get("overall_accuracy", 0)
    miou = summary.get("mean_iou", 0)

    c1, c2 = st.columns(2)
    c1.metric("Overall Accuracy", pct(accuracy))
    c2.metric("Mean IoU", pct(miou))

    # Training history is normally stored in the model-run directory.
    model_run_dir = evaluation_dir.parent

    history_candidates = [
        model_run_dir / "history.csv",
        model_run_dir / "train.csv",
    ]

    history_path = next(
        (p for p in history_candidates if p.exists()),
        None,
    )

    if history_path:
        history = pd.read_csv(history_path)

        st.subheader("Training / Validation History")

        loss_cols = [
            c for c in history.columns
            if "loss" in c.lower()
        ]

        if loss_cols:
            st.line_chart(history.set_index(history.columns[0])[loss_cols])

        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("Training history CSV was not found in the selected model-run directory.")

# ---------------------------------------------------------
# Per-Class Analysis
# ---------------------------------------------------------

elif page == "Per-Class Analysis":

    st.header("Per-Class Performance")

    if classes_df.empty:
        st.warning("test_classes.csv was not found.")
        st.stop()

    plot_df = classes_df.copy()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("IoU by Class")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(plot_df["class_name"], plot_df["iou"] * 100)
        ax.set_ylabel("IoU (%)")
        ax.set_xlabel("Class")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.subheader("F1 Score by Class")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(plot_df["class_name"], plot_df["f1"] * 100)
        ax.set_ylabel("F1 (%)")
        ax.set_xlabel("Class")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    best_iou = plot_df.loc[plot_df["iou"].idxmax()]
    worst_iou = plot_df.loc[plot_df["iou"].idxmin()]

    c1, c2 = st.columns(2)
    c1.success(
        f"Best IoU: {best_iou['class_name']} — "
        f"{best_iou['iou'] * 100:.2f}%"
    )
    c2.warning(
        f"Worst IoU: {worst_iou['class_name']} — "
        f"{worst_iou['iou'] * 100:.2f}%"
    )

# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

elif page == "Confusion Matrix":

    st.header("Confusion Matrix")

    confusion_png = evaluation_dir / "test_confusion.png"

    if confusion_png.exists():
        st.image(
            str(confusion_png),
            caption="Independent held-out test-set confusion matrix",
            use_container_width=True,
        )
    elif confusion_path.exists():
        st.dataframe(
            pd.read_csv(confusion_path),
            use_container_width=True,
        )
    else:
        st.warning("Confusion-matrix output was not found.")

# ---------------------------------------------------------
# Monthly Prototype
# ---------------------------------------------------------

elif page == "Monthly Prototype":

    st.header("January 2025 Sentinel-1 Monthly Prototype")

    st.info(
        "This section shows the real Sentinel-1 monthly prototype and "
        "the model's prediction output. The prediction is a prototype result, "
        "not independent ground truth."
    )

    # Search for likely monthly prototype images.
    all_pngs = find_pngs()

    monthly_candidates = [
        p for p in all_pngs
        if any(
            keyword in p.name.lower()
            for keyword in [
                "monthly",
                "prediction",
                "lulc",
                "january",
                "jan",
                "prototype",
                "composite",
            ]
        )
    ]

    if monthly_candidates:
        labels = [str(p.relative_to(project_dir)) for p in monthly_candidates]

        selected = st.selectbox(
            "Select monthly/prototype output",
            range(len(monthly_candidates)),
            format_func=lambda i: labels[i],
        )

        selected_path = monthly_candidates[selected]

        st.image(
            str(selected_path),
            caption=selected_path.name,
            use_container_width=True,
        )
    else:
        st.warning(
            "No monthly prototype PNG was automatically detected. "
            "The notebook outputs remain available in 05_monthly_prototype.ipynb."
        )

    st.subheader("Prototype Interpretation")

    st.markdown(
        """
        - Real Sentinel-1 VV/VH imagery was retrieved through Google Earth Engine.
        - A January 2025 monthly composite was generated.
        - The trained SmallUNet model was applied to the real raster.
        - The resulting prediction is strongly dominated by Urban/Built-up.
        - The prototype therefore demonstrates end-to-end inference, but should
          not be interpreted as a validated land-cover map.
        """
    )

# ---------------------------------------------------------
# Limitations
# ---------------------------------------------------------

elif page == "Limitations":

    st.header("Model Limitations")

    st.markdown(
        """
        ### 1. Class imbalance

        The model performs much better on some classes than on minority
        classes. Urban/Built-up, Cropland, and Water have relatively stronger
        IoU values, while several minority classes have substantially lower
        performance.

        ### 2. Prediction bias

        The held-out diagnostic showed that the model strongly favors
        Cropland and Savanna predictions and produced no predictions for
        Shrubland, Wetland, or Water within the 100-sample diagnostic subset.

        ### 3. Domain shift

        The January 2025 Lagos Sentinel-1 prototype has a different input
        distribution from the training data. This is a major reason why the
        monthly prototype should be treated as an inference demonstration
        rather than validated operational mapping.

        ### 4. Future improvement

        A stronger system would require geographically diverse training data,
        class balancing, additional temporal/geographic samples, and
        independent geographic validation.
        """
    )

    st.subheader("Key conclusion")

    st.markdown(
        """
        The project successfully demonstrates an end-to-end deep-learning
        workflow for Sentinel-1 pixel-level LULC classification, including
        real-data preprocessing, model training, independent evaluation,
        and inference on a real monthly Sentinel-1 composite.
        """
    )

st.sidebar.markdown("---")
st.sidebar.caption("DeepLearningPro • Sentinel-1 LULC • SmallUNet")
