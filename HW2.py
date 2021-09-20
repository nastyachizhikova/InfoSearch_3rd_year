import nltk
import numpy as np
import os
import pandas as pd
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Tuple


stop_words = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()
vectorizer = TfidfVectorizer(analyzer='word')


def get_all_docs() -> List[str]:
    """
    достаем все файлы с текстами субтитров и отдельно - их названия
    """
    all_docs = []
    episodes_names = []
    curr_dir = os.getcwd()

    for i in range(7):
        foldername = os.path.join(curr_dir, 'friends-data', f'Friends - season {i+1}')
        filenames = os.listdir(foldername)
        for fpath in filenames:
            episode_name = fpath.split(' - ')[1] + ' ' + fpath.split(' - ')[2].split('.')[0]
            episodes_names.append(episode_name)

            with open(os.path.join(foldername, fpath), 'r', encoding='utf-8') as f:
                episode_text = f.read()
                all_docs.append(episode_text)

    assert len(all_docs) == 165

    return all_docs, episodes_names


def unify_names(text: str) -> str:
    """
    приводим все имена в текстах к общему виду
    """
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
    """
    приводим к нижнему регистру, меняем имена,
    убираем не-слова и стоп-слова, лемматизируем
    """
    text = text.lower()
    text = unify_names(text)
    new_words = []

    for word in word_tokenize(text):
        if word.isalpha() and word not in stop_words:
            parse = morph.parse(word)[0]
            new_words.append(parse.normal_form)

    return ' '.join(new_words)


def index(corpus: List[str]) -> Tuple[np.array, list]:
    """
    индексирование
    """
    X = vectorizer.fit_transform(corpus)
    df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())

    return df, vectorizer


def get_similarities(query: str, vectorizer: TfidfVectorizer, df: pd.DataFrame) -> np.array:
    """
    препроцессим запрос так же, как тексты,
    получаем его вектор,
    считаем векторную близость
    """
    query = preprocess(query)
    query_vect = vectorizer.transform([query])
    cos = cosine_similarity(df, query_vect)

    return cos

def search(query: str, vectorizer: TfidfVectorizer, df: pd.DataFrame) -> List[str]:
    """
    основная функция поиска:
    получаем векторные близости,
    сортируем эпизоды по ней,
    возвращаем их названия
    """
    cos = get_similarities(query, vectorizer, df)
    df['similarity'] = cos
    sorted_df = df.sort_values(by=['similarity'], ascending=False)
    df.drop(columns=['similarity'], inplace=True)

    return sorted_df.index


def get_processed(docs):
    """
    если еще не препроцессили тексты, делаем это и сохраняем,
    если уже есть предобработанные, достаем из файла
    """
    curr_dir = os.getcwd()
    processed_path = os.path.join(curr_dir, 'processed_docs.txt')

    if os.path.exists(processed_path):
        with open(processed_path, encoding='utf-8') as file:
            processed_docs = eval(file.read())

            assert isinstance(processed_docs, list)
    else:
        processed_docs = [preprocess(doc) for doc in tqdm(docs)]
        with open('processed_docs.txt', 'w', encoding='utf-8') as file:
            file.write(str(processed_docs))

    return processed_docs


def main():
    docs, episodes_names = get_all_docs()
    processed_docs = get_processed(docs)
    df, vectorizer = index(processed_docs)
    df.index = episodes_names

    do_search = 1

    while do_search:
        query = input('Введите ваш запрос: ')
        answer = search(query, vectorizer, df)
        print('Первые 5 ответов по релевантности: ', '\n -', '\n - '.join(answer[:5]))

        with open('query_answer.txt', 'w', encoding='utf-8') as file:
            file.write('\n'.join(answer))

        print('Полные результаты сохранены в файле query_answer.txt')

        do_search = int(input('Введите 0, если хотите прекратить поиски, введите 1, если хотите сделать еще запрос: '))


if __name__ == '__main__':
    main()
