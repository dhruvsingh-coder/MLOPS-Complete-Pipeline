import pandas as pd
import os
import logging
import yaml
import pickle
from sklearn.naive_bayes import MultinomialNB

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(log_dir, 'model_building.log'))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def load_params(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    train = pd.read_csv('./data/processed/train_tfidf.csv')

    X = train.iloc[:, :-1]
    y = train.iloc[:, -1]

    model = MultinomialNB()
    model.fit(X, y)

    os.makedirs('models', exist_ok=True)

    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    logger.info("✅ Model Training Done")


if __name__ == "__main__":
    main()