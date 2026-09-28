import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
from dvclive import Live

# ---------------- CONFIG ----------------
MODEL_PATH = './models/model.pkl'
TEST_DATA_PATH = './data/processed/test_tfidf.csv'
METRICS_PATH = 'reports/metrics.json'

# ---------------- LOGGING ----------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, 'model_evaluation.log'))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ---------------- FUNCTIONS ----------------

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f'Model loaded from {file_path}')
        return model
    except Exception as e:
        logger.error(f'Error loading model: {e}')
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Data loaded from {file_path}')
        return df
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray):
    try:
        y_pred = clf.predict(X_test)

        # Safe probability handling
        try:
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": auc
        }

        logger.debug(f'Metrics: {metrics}')
        return metrics

    except Exception as e:
        logger.error(f'Evaluation error: {e}')
        raise


def save_metrics(metrics: dict, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug(f'Metrics saved at {file_path}')
    except Exception as e:
        logger.error(f'Error saving metrics: {e}')
        raise


# ---------------- MAIN ----------------

def main():
    try:
        model = load_model(MODEL_PATH)
        test_data = load_data(TEST_DATA_PATH)

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(model, X_test, y_test)

        # ✅ Correct dvclive logging
        with Live(save_dvc_exp=True) as live:
            for key, value in metrics.items():
                live.log_metric(key, value)

        save_metrics(metrics, METRICS_PATH)

        logger.info("✅ Model evaluation completed")

    except Exception as e:
        logger.error(f'Pipeline failed: {e}')
        print(f"Error: {e}")


if __name__ == '__main__':
    main()