import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import os
import pandas as pd
import pickle
import pymorphy2
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from typing import List

stop_words = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()


def get_value(author_rating: dict) -> int:
    '''
    функция получения value из информации об авторе
    '''
    return author_rating['value']


def get_best_answer(question_answers: dict) -> str:
    '''
    функция, выдающая текст ответа с максимальным value
    '''
    df = pd.DataFrame(question_answers)
    df['author_rating'] = df.author_rating.apply(get_value)

    return df.sort_values(by=['author_rating']).text.values[0]


def get_answers_corpus(corpus: List[dict]) -> list:
    '''
    фцнкция, получающая файл с полным корпусом и выдающая корпус только лучших ответов
    '''
    ans_corpus = []

    for question in tqdm(corpus, desc='Collecting answer data'):
        question_answers = json.loads(question)['answers']
        if question_answers:
            ans_corpus.append(get_best_answer(question_answers))

    return ans_corpus


def get_raw_data(filename: str) -> list:
    '''
    функиця, которая читет json файл и получает сырой корпус ответов (если он уже не сохранен в файле)
    '''
    curr_dir = os.getcwd()
    raw_path = os.path.join(curr_dir, 'raw_data.txt')

    if os.path.exists(raw_path):
        with open(raw_path, encoding='utf-8') as file:
            raw_data = eval(file.read())

            assert isinstance(raw_path, list)
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            corpus = list(f)[:50000]
            raw_data = get_answers_corpus(corpus)

        with open('raw_data.txt', 'w', encoding='utf-8') as file:
            file.write(str(raw_data))

    return raw_data


def preprocess(text: str) -> str:
    """
    приводим к нижнему регистру, меняем имена,
    убираем не-слова и стоп-слова, лемматизируем
    """
    text = text.lower()
    new_words = []

    for word in word_tokenize(text):
        if word.isalpha() and word not in stop_words:
            parse = morph.parse(word)[0]
            new_words.append(parse.normal_form)

    return ' '.join(new_words)


def get_corpus(filename: str) -> str:
    '''
    полная функция получения данных:
    если процесс получения сырого и обработанного корпуса уже пройден, читаем корпуса из готовых файлов,
    в противном случае обрабатываем json файл и получаем данные с нуля
    '''
    curr_dir = os.getcwd()
    processed_path = os.path.join(curr_dir, 'processed_data.txt')
    raw_path = os.path.join(curr_dir, 'raw_data.txt')

    if os.path.exists(processed_path) and os.path.exists(raw_path):
        with open(raw_path, encoding='utf-8') as file:
            raw_data = eval(file.read())

            assert isinstance(raw_data, list)

        with open(processed_path, encoding='utf-8') as file:
            processed_data = eval(file.read())

            assert isinstance(processed_data, list)
    else:
        raw_data = get_raw_data(filename)
        processed_data = [preprocess(doc) for doc in tqdm(raw_data, desc='Preprocessing files')]

        with open('processed_data.txt', 'w', encoding='utf-8') as file:
            file.write(str(processed_data))

    return processed_data, raw_data


def count_bm25(x_count_vec: sparse.csr_matrix,
               x_tf_vec: sparse.csr_matrix,
               tfidf_vectorizer: TfidfVectorizer) -> sparse.csr_matrix:
    '''
    функция для посчета значений БМ-25
    '''
    idf_vec = tfidf_vectorizer.idf_
    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()

    k = 2
    b = 0.75
    values, rows, cols = [], [], []

    for i, j in tqdm(zip(*x_tf_vec.nonzero()), desc='Counting BM-25'):
        len_doc = len_d[i]
        idf = idf_vec[j]
        tf = x_tf_vec[i,j]

        num = idf * tf * (k + 1)  # числитель
        denom = tf + (k * (1 - b + b * len_doc / avdl))  # знаменатель

        value = num / denom
        values.append(value)
        rows.append(i)
        cols.append(j)

    sparse_matrix = sparse.csr_matrix((np.array(values).squeeze(), (np.array(rows), np.array(cols))))

    return sparse_matrix


def index(corpus):
    '''
    функция индексации корпуса с помощью БМ-25 (либо читаем из файла уже посчитанные значения,
    либо считаем с нуля
    '''
    curr_dir = os.getcwd()
    indexed_path = os.path.join(curr_dir, 'indexed_data.pkl')

    if os.path.exists(indexed_path):
        with open('indexed_data.pkl', 'rb') as f:
            indexed_data = pickle.load(f)

        with open('count_vectorizer.pkl', 'rb') as f:
            count_vectorizer = pickle.load(f)

    else:
        count_vectorizer = CountVectorizer()
        tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

        x_count_vec = count_vectorizer.fit_transform(corpus)
        x_tf_vec = tf_vectorizer.fit_transform(corpus)
        x_tfidf_vec = tfidf_vectorizer.fit_transform(corpus)

        assert isinstance(x_tf_vec, sparse.csr.csr_matrix)

        indexed_data = count_bm25(x_count_vec, x_tf_vec, tfidf_vectorizer)

        with open('indexed_data.pkl', 'wb') as f:
            pickle.dump(indexed_data, f)

        with open('count_vectorizer.pkl', 'wb') as f:
            pickle.dump(count_vectorizer, f)

    return indexed_data, count_vectorizer


def get_similarity_scores(query: str, count_vectorizer: CountVectorizer, indexed_data: sparse.csr_matrix) -> np.array:
    '''
    функция подсчета близости query и индексированного корпуса
    '''
    query_vec = count_vectorizer.transform([query])
    scores = indexed_data.dot(query_vec.T)

    return scores.toarray().squeeze()


def search(query: str,
           count_vectorizer: CountVectorizer,
           indexed_data: sparse.csr_matrix,
           ans_corpus: np.array) -> np.array:
    '''
    полная функция запроса: принимает запрос, выдает отсортированные по релевантности выдачи из корпуса
    '''
    scores = get_similarity_scores(query, count_vectorizer, indexed_data)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    answer = ans_corpus[sorted_scores_indx.ravel()[:1000]]  # здесь моя цпу не выдерживает, можно убрать срез

    return answer


def main():
    corpus, raw_corpus = get_corpus('questions_about_love.jsonl')
    indexed_data, count_vectorizer = index(corpus)
    
    do_search = 1

    while do_search:
        query = input('Введите ваш запрос: ')
        answer = search(preprocess(query), count_vectorizer, indexed_data, np.array(raw_corpus))
        print('Первые 5 ответов по релевантности: ', '\n -', '\n - '.join(answer[:5]))

        with open('query_answer.txt', 'w', encoding='utf-8') as file:
            file.write('\n'.join(answer))

        print('Полные результаты (10,000) сохранены в файле query_answer.txt')

        do_search = int(input('Введите 0, если хотите прекратить поиски, введите 1, если хотите сделать еще запрос: '))


if __name__ == '__main__':
    main()
