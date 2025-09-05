import shap
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path
from src.config.config import CONFIG  # central config

class ModelExplainer:
    def __init__(self, output_dir=None):
        # Use artifacts_dir from config if not provided
        self.output_dir = Path(output_dir or CONFIG["artifacts_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("📊 ModelExplainer initialized.")
        self.plots = {}  # store matplotlib figures here

    def explain_model(self, model, X, feature_names, model_name="Model", dataset_name="Dataset"):
        logging.info(f"🔍 Explaining {model_name} on {dataset_name} with SHAP...")
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # --- Summary Plot ---
        fig_summary = plt.figure()
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        summary_path = self.output_dir / f"{model_name}_{dataset_name}_shap_summary.png"
        plt.savefig(summary_path, bbox_inches="tight")
        plt.close(fig_summary)
        logging.info(f"✅ SHAP summary plot saved to {summary_path}")
        self.plots[f"{model_name}_{dataset_name}_shap_summary"] = fig_summary

        # --- Force Plot (first instance) ---
        try:
            force_path = self.output_dir / f"{model_name}_{dataset_name}_shap_force.html"
            force_plot = shap.plots.force(shap_values[0], matplotlib=False)
            shap.save_html(str(force_path), force_plot)
            logging.info(f"✅ SHAP force plot saved to {force_path}")
        except Exception as e:
            logging.error(f"❌ Failed to save SHAP force plot for {model_name} on {dataset_name}: {e}")
            force_path = None

        # Return file paths and plots dict so pipeline can log them
        return str(summary_path), str(force_path) if force_path else None, self.plots
