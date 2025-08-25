import shap
import matplotlib.pyplot as plt
import logging
import os

class ModelExplainer:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info("📊 ModelExplainer initialized.")

    def explain_model(self, model, X, feature_names, model_name="Model", dataset_name="Dataset"):
        logging.info(f"🔍 Explaining {model_name} on {dataset_name} with SHAP...")
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        summary_path = os.path.join(self.output_dir, f"{model_name}_{dataset_name}_shap_summary.png")
        plt.savefig(summary_path)
        logging.info(f"✅ SHAP summary plot saved to {summary_path}")

        # Force Plot (first instance)
        force_plot_html = shap.plots.force(shap_values[0], matplotlib=False)
        force_path = os.path.join(self.output_dir, f"{model_name}_{dataset_name}_shap_force.html")
        with open(force_path, "w") as f:
            f.write(force_plot_html.data)
        logging.info(f"✅ SHAP force plot saved to {force_path}")

        return summary_path, force_path
