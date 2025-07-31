import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier # Using LightGBM as the powerful ensemble model
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, f1_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split # For internal testing if needed
import os
import sys # Import the sys module

# Dynamically add the 'src' directory to sys.path
# This allows importing modules from 'src' when pipeline.py is in 'src/models/'
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir) # Insert at the beginning to prioritize local modules

# --- DIAGNOSTIC PRINTS ---
# These prints will help you verify the paths Python is using.
print(f"DEBUG: Current script directory: {current_dir}")
print(f"DEBUG: Calculated src directory to add: {src_dir}")
print(f"DEBUG: Current sys.path after modification: {sys.path}")
# --- END DIAGNOSTIC PRINTS ---
from data_cleaner import DataCleaner
from data_loader import DataLoader
from data_transformer import DataTransformer

class ModelTrainer:
    """
    A class to build, train, and evaluate machine learning models for fraud detection.
    """
    def __init__(self):
        """
        Initializes the ModelTrainer with chosen models.
        """
        print("ModelTrainer initialized.")
        # Initialize models
        # Logistic Regression: Simple, interpretable baseline
        self.logistic_regression_model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
        # LightGBM: Powerful ensemble model, good for imbalanced data and performance
        # Using 'scale_pos_weight' or 'is_unbalance' is common for imbalanced data,
        # but since we are using SMOTE on the training data, we might not need it explicitly here.
        # However, 'class_weight' can still be beneficial.
        self.lgbm_model = LGBMClassifier(random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31, class_weight='balanced')

    def _plot_confusion_matrix(self, y_true, y_pred, model_name, dataset_name):
        """
        Plots the confusion matrix for model predictions.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def _plot_precision_recall_curve(self, y_true, y_scores, model_name, dataset_name):
        """
        Plots the Precision-Recall curve.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auc_pr = auc(recall, precision)

        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {auc_pr:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name} on {dataset_name}')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.show()


    def evaluate_model(self, model, X_test, y_test, model_name, dataset_name):
        """
        Evaluates a trained model using appropriate metrics for imbalanced data.

        Args:
            model: The trained machine learning model.
            X_test (np.array or pd.DataFrame): Test features.
            y_test (pd.Series): True labels for the test set.
            model_name (str): Name of the model for logging/plotting.
            dataset_name (str): Name of the dataset for logging/plotting.

        Returns:
            dict: A dictionary containing evaluation metrics (F1-Score, AUC-PR).
        """
        print(f"\n--- Evaluating {model_name} on {dataset_name} ---")

        y_pred = model.predict(X_test)
        # For AUC-PR, we need probability scores for the positive class (class 1)
        y_scores = model.predict_proba(X_test)[:, 1]

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # F1-Score
        f1 = f1_score(y_test, y_pred)
        print(f"F1-Score: {f1:.4f}")

        # AUC-PR (Average Precision Score)
        auc_pr = average_precision_score(y_test, y_scores)
        print(f"AUC-PR: {auc_pr:.4f}")

        # Plot Confusion Matrix
        self._plot_confusion_matrix(y_test, y_pred, model_name, dataset_name)

        # Plot Precision-Recall Curve
        self._plot_precision_recall_curve(y_test, y_scores, model_name, dataset_name)

        return {'f1_score': f1, 'auc_pr': auc_pr}

    def train_and_evaluate(self, model, X_train, y_train, X_test, y_test, model_name, dataset_name):
        """
        Trains a given model and then evaluates it.

        Args:
            model: The machine learning model to train.
            X_train (np.array or pd.DataFrame): Training features.
            y_train (pd.Series): True labels for the training set.
            X_test (np.array or pd.DataFrame): Test features.
            y_test (pd.Series): True labels for the test set.
            model_name (str): Name of the model for logging/plotting.
            dataset_name (str): Name of the dataset for logging/plotting.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        print(f"\n--- Training {model_name} on {dataset_name} ---")
        model.fit(X_train, y_train)
        print(f"{model_name} training complete.")
        metrics = self.evaluate_model(model, X_test, y_test, model_name, dataset_name)
        return metrics

    def run_all_models(self, prepared_data):
        """
        Runs the full model training and evaluation pipeline on both datasets.

        Args:
            prepared_data (dict): A dictionary containing the prepared training and testing sets
                                  for both datasets, as returned by DataTransformer.transform_data_for_ml.

        Returns:
            dict: A nested dictionary containing evaluation metrics for all models on both datasets.
        """
        print("\n--- Starting Model Building and Training ---")
        all_metrics = {}

        # --- Fraud Data Models ---
        print("\n--- Processing Fraud_Data.csv Models ---")
        X_train_fraud_resampled = prepared_data['X_train_fraud_resampled']
        y_train_fraud_resampled = prepared_data['y_train_fraud_resampled']
        X_test_fraud_processed = prepared_data['X_test_fraud_processed']
        y_test_fraud = prepared_data['y_test_fraud']

        # Logistic Regression for Fraud Data
        lr_fraud_metrics = self.train_and_evaluate(
            self.logistic_regression_model,
            X_train_fraud_resampled, y_train_fraud_resampled,
            X_test_fraud_processed, y_test_fraud,
            "Logistic Regression", "Fraud_Data.csv"
        )
        all_metrics['lr_fraud'] = lr_fraud_metrics

        # LightGBM for Fraud Data
        lgbm_fraud_metrics = self.train_and_evaluate(
            self.lgbm_model,
            X_train_fraud_resampled, y_train_fraud_resampled,
            X_test_fraud_processed, y_test_fraud,
            "LightGBM", "Fraud_Data.csv"
        )
        all_metrics['lgbm_fraud'] = lgbm_fraud_metrics

        # --- Credit Card Data Models ---
        print("\n\n--- Processing creditcard.csv Models ---")
        X_train_creditcard_resampled = prepared_data['X_train_creditcard_resampled']
        y_train_creditcard_resampled = prepared_data['y_train_creditcard_resampled']
        X_test_creditcard_processed = prepared_data['X_test_creditcard_processed']
        y_test_creditcard = prepared_data['y_test_creditcard']

        # Logistic Regression for Credit Card Data
        lr_cc_metrics = self.train_and_evaluate(
            self.logistic_regression_model, # Re-using the same model instance, it will be refitted
            X_train_creditcard_resampled, y_train_creditcard_resampled,
            X_test_creditcard_processed, y_test_creditcard,
            "Logistic Regression", "creditcard.csv"
        )
        all_metrics['lr_creditcard'] = lr_cc_metrics

        # LightGBM for Credit Card Data
        lgbm_cc_metrics = self.train_and_evaluate(
            self.lgbm_model, # Re-using the same model instance, it will be refitted
            X_train_creditcard_resampled, y_train_creditcard_resampled,
            X_test_creditcard_processed, y_test_creditcard,
            "LightGBM", "creditcard.csv"
        )
        all_metrics['lgbm_creditcard'] = lgbm_cc_metrics

        print("\n--- Model Training and Evaluation Complete ---")
        print("\n--- Summary of All Model Metrics ---")
        for key, metrics in all_metrics.items():
            print(f"{key}: F1-Score={metrics['f1_score']:.4f}, AUC-PR={metrics['auc_pr']:.4f}")

        self._justify_best_model(all_metrics)

        return all_metrics

    def _justify_best_model(self, all_metrics):
        """
        Provides a justification for the best model based on AUC-PR and F1-Score.
        """
        print("\n--- Justification for Best Model ---")

        # For Fraud_Data.csv
        print("\n--- Fraud_Data.csv ---")
        lr_fraud_f1 = all_metrics['lr_fraud']['f1_score']
        lr_fraud_auc_pr = all_metrics['lr_fraud']['auc_pr']
        lgbm_fraud_f1 = all_metrics['lgbm_fraud']['f1_score']
        lgbm_fraud_auc_pr = all_metrics['lgbm_fraud']['auc_pr']

        print(f"Logistic Regression (Fraud): F1={lr_fraud_f1:.4f}, AUC-PR={lr_fraud_auc_pr:.4f}")
        print(f"LightGBM (Fraud): F1={lgbm_fraud_f1:.4f}, AUC-PR={lgbm_fraud_auc_pr:.4f}")

        if lgbm_fraud_auc_pr > lr_fraud_auc_pr and lgbm_fraud_f1 > lr_fraud_f1:
            print("For Fraud_Data.csv, LightGBM appears to be the better model.")
            print("Justification: LightGBM generally outperforms Logistic Regression in both F1-Score and AUC-PR.")
            print("This indicates it has a better balance of precision and recall, and a higher average precision across all thresholds, which is crucial for detecting rare fraudulent transactions.")
            print("Ensemble methods like LightGBM can capture more complex non-linear relationships in the data compared to a linear model like Logistic Regression.")
        else:
            print("For Fraud_Data.csv, Logistic Regression might be preferred for its simplicity and interpretability if its performance is competitive.")
            print("However, LightGBM is generally expected to perform better for complex fraud detection tasks.")


        # For creditcard.csv
        print("\n--- creditcard.csv ---")
        lr_cc_f1 = all_metrics['lr_creditcard']['f1_score']
        lr_cc_auc_pr = all_metrics['lr_creditcard']['auc_pr']
        lgbm_cc_f1 = all_metrics['lgbm_creditcard']['f1_score']
        lgbm_cc_auc_pr = all_metrics['lgbm_creditcard']['auc_pr']

        print(f"Logistic Regression (Credit Card): F1={lr_cc_f1:.4f}, AUC-PR={lr_cc_auc_pr:.4f}")
        print(f"LightGBM (Credit Card): F1={lgbm_cc_f1:.4f}, AUC-PR={lgbm_cc_auc_pr:.4f}")

        if lgbm_cc_auc_pr > lr_cc_auc_pr and lgbm_cc_f1 > lr_cc_f1:
            print("For creditcard.csv, LightGBM appears to be the better model.")
            print("Justification: Similar to Fraud_Data.csv, LightGBM shows superior performance in both F1-Score and AUC-PR.")
            print("The anonymized V-features in the creditcard dataset likely contain complex interactions that LightGBM, as a gradient boosting model, is well-suited to learn.")
            print("Given the extreme imbalance in this dataset, maximizing AUC-PR and F1-Score is paramount, and LightGBM typically excels in these areas.")
        else:
            print("For creditcard.csv, Logistic Regression might be preferred for its simplicity and interpretability if its performance is competitive.")
            print("However, LightGBM is generally expected to perform better for complex fraud detection tasks.")


if __name__ == '__main__':
    # --- Standalone Example Usage for ModelTrainer ---
    # This block simulates receiving prepared_data from DataTransformer.
    # For a full run, use main_preprocessing.py.

    print("--- Running ModelTrainer Standalone Example ---")

    # Simulate loading and cleaning data (as done in main_preprocessing.py)
    data_loader = DataLoader()
    dataset_paths = {
        'fraud_data': '../data/processed/fraud_data_cleaned.csv',  # Adjust paths if needed for your local setup
        'ip_to_country': '../data/processed/ip_to_country_cleaned.csv',
        'creditcard_data': '../data/processed/creditcard_data_cleaned.csv'
    }
    loaded_dfs = data_loader.load_data(dataset_paths)

    data_cleaner = DataCleaner()
    cleaned_dfs = data_cleaner.clean_all_datasets(loaded_dfs)

    fraud_df_cleaned = cleaned_dfs.get('fraud_data')
    ip_df_cleaned = cleaned_dfs.get('ip_to_country')
    cc_df_cleaned = cleaned_dfs.get('creditcard_data')

    if fraud_df_cleaned is None or ip_df_cleaned is None or cc_df_cleaned is None:
        print("Error: Required cleaned data is missing. Cannot proceed with model training.")
    else:
        # Simulate data transformation (as done in main_preprocessing.py)
        data_transformer = DataTransformer()
        prepared_data = data_transformer.transform_data_for_ml(
            fraud_df_cleaned,
            ip_df_cleaned,
            cc_df_cleaned
        )

        # Run the ModelTrainer
        model_trainer = ModelTrainer()
        all_model_metrics = model_trainer.run_all_models(prepared_data)

        print("\n--- Final Model Metrics from Standalone Run ---")
        print(all_model_metrics)
