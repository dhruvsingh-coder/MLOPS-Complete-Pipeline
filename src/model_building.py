import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

# ---------------- CONFIG (INBUILT) ----------------
N_ESTIMATORS = 100
RANDOM_STATE = 42
TRAIN_DATA_PATH = './data/processed/train_tfidf.csv'
MODEL_PATH = 'models/model.pkl'

# ---------------- LOGGING ----------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, 'model_building.log'))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ---------------- FUNCTIONS ----------------

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Data loaded from {file_path}, shape: {df.shape}')
        return df
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Mismatch in X and y sizes")

        logger.debug("Initializing RandomForest")

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE
        )

        logger.debug("Training started...")
        clf.fit(X_train, y_train)
        logger.debug("Training completed")

        return clf

    except Exception as e:
        logger.error(f'Training error: {e}')
        raise


def save_model(model, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(model, f)

        logger.debug(f'Model saved at {file_path}')

    except Exception as e:
        logger.error(f'Error saving model: {e}')
        raise


# ---------------- MAIN ----------------

def main():
    try:
        train_data = load_data(TRAIN_DATA_PATH)

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        model = train_model(X_train, y_train)

        save_model(model, MODEL_PATH)

        logger.info("✅ Model training completed successfully")

    except Exception as e:
        logger.error(f'Pipeline failed: {e}')
        print(f"Error: {e}")


if __name__ == '__main__':
    main()