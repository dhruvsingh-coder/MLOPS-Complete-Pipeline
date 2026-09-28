import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import joblib

# ---------------- CONFIG (INBUILT) ----------------
MAX_FEATURES = 500   # ✅ you control from here
TRAIN_PATH = './data/interim/train_processed.csv'
TEST_PATH = './data/interim/test_processed.csv'

# ---------------- LOGGING ----------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, 'feature_engineering.log'))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ---------------- FUNCTIONS ----------------

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug(f'Data loaded from {file_path}')
        return df
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame):
    try:
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)

        X_train = train_data['text']
        y_train = train_data['target']
        X_test = test_data['text']
        y_test = test_data['target']

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_vec.toarray())
        train_df['target'] = y_train.values

        test_df = pd.DataFrame(X_test_vec.toarray())
        test_df['target'] = y_test.values

        # ✅ Save vectorizer (VERY IMPORTANT)
        os.makedirs("models", exist_ok=True)
        joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

        logger.debug('TF-IDF applied successfully')

        return train_df, test_df

    except Exception as e:
        logger.error(f'TF-IDF error: {e}')
        raise


def save_data(df: pd.DataFrame, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f'Data saved at {file_path}')
    except Exception as e:
        logger.error(f'Error saving data: {e}')
        raise


# ---------------- MAIN ----------------

def main():
    try:
        train_data = load_data(TRAIN_PATH)
        test_data = load_data(TEST_PATH)

        train_df, test_df = apply_tfidf(train_data, test_data)

        save_data(train_df, './data/processed/train_tfidf.csv')
        save_data(test_df, './data/processed/test_tfidf.csv')

        logger.info("✅ Feature engineering completed")

    except Exception as e:
        logger.error(f'Pipeline failed: {e}')
        print(f"Error: {e}")


if __name__ == '__main__':
    main()