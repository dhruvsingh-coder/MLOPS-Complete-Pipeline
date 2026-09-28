import pandas as pd
import os
import logging
import nltk
import yaml
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(log_dir, 'data_preprocessing.log'))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def load_params(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def transform(text):
    ps = PorterStemmer()
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stopwords.words('english')]
    words = [ps.stem(w) for w in words]
    return " ".join(words)


def preprocess(df):
    df.drop_duplicates(inplace=True)
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    df['text'] = df['text'].apply(transform)
    return df


def main():
    train = pd.read_csv('./data/raw/train.csv')
    test = pd.read_csv('./data/raw/test.csv')

    train = preprocess(train)
    test = preprocess(test)

    os.makedirs('./data/interim', exist_ok=True)

    train.to_csv('./data/interim/train_processed.csv', index=False)
    test.to_csv('./data/interim/test_processed.csv', index=False)

    logger.info("✅ Preprocessing Done")


if __name__ == "__main__":
    main()