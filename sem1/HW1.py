import nltk
import numpy as np
import os
import pandas as pd
import pymorphy2
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Tuple


stop_words = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()
vectorizer = CountVectorizer(analyzer='word')


def get_all_docs() -> List[str]:
    all_docs = []
    curr_dir = os.getcwd()

    for i in range(7):
        foldername = os.path.join(curr_dir, 'friends-data', f'Friends - season {i+1}')
        filenames = os.listdir(foldername)
        for fpath in filenames:
            with open(os.path.join(foldername, fpath), 'r', encoding='utf-8') as f:
                episode_text = f.read()
                all_docs.append(episode_text)

    assert len(all_docs) == 165

    return all_docs


def unify_names(text: str) -> str:
    names_dict = {'джоуи': 'джо',
                 'джои': 'джо',
                 'фибс': 'фиби',
                 'мон': 'моника',
                 'чэндлер': 'чендлер',
                 'чен': 'чендлер',
                 'рэйчел': 'рейчел',
                 'рейч': 'рейчел'}

    for item, value in names_dict.items():
        text = text.replace(item, value)

    return text


def preprocess(text: str) -> str:
    text = text.lower()
    text = unify_names(text)
    new_words = []

    for word in word_tokenize(text):
        if word.isalpha() and word not in stop_words:
            parse = morph.parse(word)[0]
            new_words.append(parse.normal_form)

    return ' '.join(new_words)


def index(corpus: List[str]) -> Tuple[np.array, list]:
    X = vectorizer.fit_transform(corpus)
    df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())

    return df


def print_results(df: pd.DataFrame()):
    total = df.sum()
    max_freq = total.idxmax()
    print('Самое частоупотребимое слово: ', max_freq)

    min_freq = total.idxmin()
    print('Самое редкоупотребимое слово: ', min_freq)

    words_from_all_docs = df.T.loc[(df.T!=0).all(axis=1)].index.values
    print('Набор слов, который есть во всех документах коллекции: ', words_from_all_docs)

    df_names = df[['моника', 'росс', 'фиби', 'чендлера', 'джо', 'рейчел']].sum()
    max_name = df_names.idxmax()
    print('Имя, которое встречалось чаще всего: ', max_name)


def main():
    docs = get_all_docs()
    all_docs = [preprocess(doc) for doc in tqdm(docs)]
    df = index(all_docs)
    print_results(df)


if __name__ == '__main__':
    main()
