import pandas as pd
import os
import logging
import yaml
from sklearn.model_selection import train_test_split

# Logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(log_dir, 'data_ingestion.log'))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def load_params(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_data(url):
    return pd.read_csv(url, encoding='latin-1')


def preprocess(df):
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore', inplace=True)
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
    return df


def save_data(train, test):
    path = './data/raw'
    os.makedirs(path, exist_ok=True)
    train.to_csv(f"{path}/train.csv", index=False)
    test.to_csv(f"{path}/test.csv", index=False)


def main():
    params = load_params('params.yaml')
    test_size = params['data_ingestion']['test_size']

    url = "https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"

    df = load_data(url)
    df = preprocess(df)

    train, test = train_test_split(df, test_size=test_size, random_state=42)

    save_data(train, test)

    logger.info("✅ Data Ingestion Done")


if __name__ == "__main__":
    main()