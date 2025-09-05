from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    f1_score, average_precision_score, confusion_matrix,
    precision_recall_curve, auc, precision_score, recall_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class ModelTrainer:
    def __init__(self):
        logging.info("🧠 ModelTrainer initialized.")
        self.logistic_regression_model = LogisticRegression(
            solver='liblinear', class_weight='balanced', random_state=42
        )
        self.lgbm_model = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, class_weight='balanced', random_state=42
        )
        self.plots = {}  # store generated figures here

    def train_and_evaluate(self, model, X_train, y_train, X_test, y_test, model_name, dataset_name):
        logging.info(f"🚀 Training {model_name} on {dataset_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        auc_pr = average_precision_score(y_test, y_scores)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)
        logging.info(
            f"{model_name} F1: {f1:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}, AUC-PR: {auc_pr:.4f}"
        )

        # Store plots instead of showing them
        self._plot_confusion_matrix(y_test, y_pred, model_name, dataset_name)
        self._plot_precision_recall_curve(y_test, y_scores, model_name, dataset_name)

        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'auc_pr': auc_pr
        }, model

    def _plot_confusion_matrix(self, y_true, y_pred, model_name, dataset_name):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{model_name} Confusion Matrix ({dataset_name})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        self.plots[f"{model_name}_{dataset_name}_confusion_matrix"] = fig

    def _plot_precision_recall_curve(self, y_true, y_scores, model_name, dataset_name):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f"AUC-PR: {auc(recall, precision):.2f}")
        ax.set_title(f"{model_name} Precision-Recall Curve ({dataset_name})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.grid(True)
        self.plots[f"{model_name}_{dataset_name}_precision_recall_curve"] = fig

    def run_all_models(self, prepared_data):
        results = {}
        models = {}
        model_dataset_map = {}

        for dataset in ['fraud', 'creditcard']:
            data_dict = prepared_data[dataset]

            # Create fresh model instances for each dataset
            lr_model = LogisticRegression(
                solver='liblinear', class_weight='balanced', random_state=42
            )
            lgbm_model = LGBMClassifier(
                n_estimators=500, learning_rate=0.05,
                class_weight='balanced', random_state=42
            )

            for model_name, model in [('lr', lr_model), ('lgbm', lgbm_model)]:
                X_train = data_dict[f'X_train_{dataset}_resampled']
                y_train = data_dict[f'y_train_{dataset}_resampled']
                X_test = data_dict[f'X_test_{dataset}']
                y_test = data_dict[f'y_test_{dataset}']

                metrics, trained_model = self.train_and_evaluate(
                    model, X_train, y_train, X_test, y_test,
                    model_name.upper(), f"{dataset}.csv"
                )

                key = f'{model_name}_{dataset}'
                results[key] = metrics
                models[key] = trained_model
                model_dataset_map[key] = dataset

        # Return plots so pipeline can save/log them
        return results, models, model_dataset_map, self.plots
