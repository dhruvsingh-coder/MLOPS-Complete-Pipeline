import pandas as pd
import os
import logging
import yaml
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- LOGGING ----------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(log_dir, 'feature_engineering.log'))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

# ---------------- PARAMS ----------------
def load_params(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ---------------- MAIN ----------------
def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']

        # Load data
        train = pd.read_csv('./data/interim/train_processed.csv')
        test = pd.read_csv('./data/interim/test_processed.csv')

        # 🔥 FIX 1: Handle NaN values
        train['text'] = train['text'].fillna('')
        test['text'] = test['text'].fillna('')

        # 🔥 FIX 2: Remove empty rows (important)
        train = train[train['text'].str.strip() != '']
        test = test[test['text'].str.strip() != '']

        logger.debug(f"Train shape after cleaning: {train.shape}")
        logger.debug(f"Test shape after cleaning: {test.shape}")

        # TF-IDF
        tfidf = TfidfVectorizer(max_features=max_features)

        X_train = tfidf.fit_transform(train['text'])
        X_test = tfidf.transform(test['text'])

        # Convert to DataFrame
        train_df = pd.DataFrame(X_train.toarray())
        train_df['target'] = train['target'].values

        test_df = pd.DataFrame(X_test.toarray())
        test_df['target'] = test['target'].values

        # Save vectorizer
        os.makedirs('models', exist_ok=True)
        joblib.dump(tfidf, 'models/tfidf.pkl')

        # Save processed data
        os.makedirs('./data/processed', exist_ok=True)
        train_df.to_csv('./data/processed/train_tfidf.csv', index=False)
        test_df.to_csv('./data/processed/test_tfidf.csv', index=False)

        logger.info("✅ Feature Engineering Completed Successfully")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()