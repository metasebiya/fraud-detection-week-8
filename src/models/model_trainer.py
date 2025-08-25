from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score, average_precision_score, confusion_matrix, precision_recall_curve, auc, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class ModelTrainer:
    def __init__(self):
        logging.info("🧠 ModelTrainer initialized.")
        self.logistic_regression_model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
        self.lgbm_model = LGBMClassifier(n_estimators=500, learning_rate=0.05, class_weight='balanced', random_state=42)

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
            f"{model_name} F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}, AUC-PR: {auc_pr:.4f}")

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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{model_name} Confusion Matrix ({dataset_name})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def _plot_precision_recall_curve(self, y_true, y_scores, model_name, dataset_name):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, label=f"AUC-PR: {auc(recall, precision):.2f}")
        plt.title(f"{model_name} Precision-Recall Curve ({dataset_name})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_all_models(self, prepared_data):
        results = {}
        models = {}

        for dataset in ['fraud', 'creditcard']:
            data_dict = prepared_data[dataset]  # nested dict
            for model_name, model in [('lr', self.logistic_regression_model), ('lgbm', self.lgbm_model)]:
                X_train = data_dict[f'X_train_{dataset}_resampled']
                y_train = data_dict[f'y_train_{dataset}_resampled']
                X_test = data_dict[f'X_test_{dataset}']
                y_test = data_dict[f'y_test_{dataset}']

                metrics, trained_model = self.train_and_evaluate(
                    model, X_train, y_train, X_test, y_test,
                    model_name.upper(), f"{dataset}.csv"
                )
                results[f'{model_name}_{dataset}'] = metrics
                models[f'{model_name}_{dataset}'] = trained_model

        return results, models
