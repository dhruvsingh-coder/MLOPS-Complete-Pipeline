import pandas as pd
import pickle
import json
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dvclive import Live

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(log_dir, 'model_evaluation.log'))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def main():
    model = pickle.load(open('models/model.pkl', 'rb'))
    test = pd.read_csv('./data/processed/test_tfidf.csv')

    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba)
    }

    os.makedirs('reports', exist_ok=True)

    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    with Live(save_dvc_exp=True) as live:
        for k, v in metrics.items():
            live.log_metric(k, v)

    logger.info("✅ Evaluation Done")


if __name__ == "__main__":
    main()