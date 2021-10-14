import gensim
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
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from typing import List, Tuple

stop_words = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()

m = './araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
fasttext_model = gensim.models.KeyedVectors.load(m)

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
model.cuda()


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


def get_answers_questions_corpus(corpus: List[dict]) -> Tuple[list, list]:
    '''
    фцнкция, получающая файл с полным корпусом и выдающая корпус только лучших ответов
    '''
    ans_corpus = []
    quest_corpus = []

    for question in tqdm(corpus, desc='Collecting raw data'):
        question_answers = json.loads(question)['answers']
        question = json.loads(question)['question']
        if question_answers and question:
            ans_corpus.append(get_best_answer(question_answers))
            quest_corpus.append(question)

    return (ans_corpus, quest_corpus)


def get_raw_data(filename: str) -> list:
    '''
    функиця, которая читет json файл и получает сырой корпус ответов (если он уже не сохранен в файле)
    '''
    curr_dir = os.getcwd()
    raw_path = os.path.join(curr_dir, 'raw_data.txt')

    if os.path.exists(raw_path):
        with open(raw_path, encoding='utf-8') as file:
            raw_data = eval(file.read())

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


        with open(processed_path, encoding='utf-8') as file:
            processed_data = eval(file.read())

    else:
        raw_ans_data, raw_quest_data = get_answers_questions_corpus(filename)
        processed_ans_data = [preprocess(doc) for doc in tqdm(raw_ans_data, desc='Preprocessing answers')]
        processed_quest_data = [preprocess(doc) for doc in tqdm(raw_quest_data, desc='Preprocessing questions')]
        
        raw_data = (raw_ans_data, raw_quest_data) 
        processed_data = (processed_ans_data, processed_quest_data)

        with open('processed_data.txt', 'w', encoding='utf-8') as file:
            file.write(str(processed_data))

    return processed_data, raw_data


def index_count(corpus: List[str], count_vectorizer=None, corpus_type='answer') \
        -> Tuple[CountVectorizer, sparse.csr_matrix]:
    """
    индексирование Count-Vect
    """
    curr_dir = os.getcwd()
    filename = f'indexed_{corpus_type}_count.pkl'
    indexed_path = os.path.join(curr_dir, filename)

    if os.path.exists(indexed_path):
        with open(filename, 'rb') as file_corpus, open('count_vect.pkl', 'rb') as file_vect:
            X = pickle.load(file_corpus)
            count_vectorizer = pickle.load(file_vect)
            
    elif corpus_type == 'answer':
        count_vectorizer = CountVectorizer()
        X = count_vectorizer.fit_transform(corpus)

        with open(filename, 'wb') as file_corpus, open('count_vect.pkl', 'wb') as file_vect:
            pickle.dump(X, file_corpus)
            pickle.dump(count_vectorizer, file_vect)
            
    elif corpus_type == 'question':
        X = count_vectorizer.transform(corpus)
        
        with open(filename, 'wb') as file_corpus:
            pickle.dump(X, file_corpus)

    return count_vectorizer, X


def index_tfidf(corpus: List[str], tf_idf_vectorizer=None, corpus_type='answer') \
        -> Tuple[TfidfVectorizer, sparse.csr_matrix]:
    """
    индексирование Tf-Idf
    """
    curr_dir = os.getcwd()
    filename = f'indexed_{corpus_type}_tfidf.pkl'
    indexed_path = os.path.join(curr_dir, filename)

    if os.path.exists(indexed_path):
        with open(filename, 'rb') as file_corpus, open('tf_idf_vect.pkl', 'rb') as file_vect:
            X = pickle.load(file_corpus)
            tf_idf_vectorizer = pickle.load(file_vect)

    elif corpus_type == 'answer':
        tf_idf_vectorizer = TfidfVectorizer()
        X = tf_idf_vectorizer.fit_transform(corpus)

        with open(filename, 'wb') as file_corpus, open('tf_idf_vect.pkl', 'wb') as file_vect:
            pickle.dump(X, file_corpus)
            pickle.dump(tf_idf_vectorizer, file_vect)
    
    elif corpus_type == 'question':
        X = tf_idf_vectorizer.transform(corpus)
        
        with open(filename, 'wb') as file_corpus:
            pickle.dump(X, file_corpus)

    return tf_idf_vectorizer, X


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


def index_bm25(corpus: List[str]):
    """
    функция индексации корпуса с помощью БМ-25 (либо читаем из файла уже посчитанные значения,
    либо считаем с нуля)
    """
    curr_dir = os.getcwd()
    filename = f'indexed_bm25.pkl'
    indexed_path = os.path.join(curr_dir, filename)

    if os.path.exists(indexed_path):
        with open(filename, 'rb') as file_corpus:
            indexed_data = pickle.load(file_corpus)

    else:
        count_vectorizer = CountVectorizer()
        tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

        x_count_vec = count_vectorizer.fit_transform(corpus)
        x_tf_vec = tf_vectorizer.fit_transform(corpus)
        x_tfidf_vec = tfidf_vectorizer.fit_transform(corpus)

        assert isinstance(x_tf_vec, sparse.csr.csr_matrix)

        indexed_data = count_bm25(x_count_vec, x_tf_vec, tfidf_vectorizer)

        with open(filename, 'wb') as file_corpus:
            pickle.dump(indexed_data, file_corpus)

    return indexed_data


def get_text_vector(text: str) -> np.array:
    """
    функция получения вектора fasttext для одного текста
    """
    vectors = []
    for word in text.split():
        vectors.append(fasttext_model[word])
     
    if vectors:
        text_vector = np.mean(np.array(vectors), 0)
    else:
        text_vector = np.zeros(300)

    return text_vector


def index_fasttext(corpus: List[str], corpus_type='answer') -> np.array:
    """
    индексирование fasttext
    """
    curr_dir = os.getcwd()
    filename = f'indexed_{corpus_type}_fasttext.pkl'
    indexed_path = os.path.join(curr_dir, filename)

    if os.path.exists(indexed_path):
        with open(filename, 'rb') as file_corpus:
            X = pickle.load(file_corpus)

    else:
        X = np.array([get_text_vector(text) for text in corpus])

        with open(filename, 'wb') as file_corpus:
            pickle.dump(X, file_corpus)

    return X


def embed_bert_cls(text: str, model: AutoModel, tokenizer: AutoTokenizer) -> np.array:
    """
    функция получения эбеддинга берта для 1 текста
    """
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    
    return embeddings[0].cpu().numpy()


def index_bert(corpus: List[str], corpus_type='answer') -> np.array:
    """
    индексирование корпуса бертом
    """
    curr_dir = os.getcwd()
    filename = f'indexed_{corpus_type}_bert.pkl'
    indexed_path = os.path.join(curr_dir, filename)

    if os.path.exists(indexed_path):
        with open(filename, 'rb') as file_corpus:
            X = pickle.load(file_corpus)

    else:
        X = np.array([embed_bert_cls(text, model, tokenizer) for text in tqdm(corpus, desc='Indexing corpus with BERT')])

        with open(filename, 'wb') as file_corpus:
            pickle.dump(X, file_corpus)

    return X


def index_all_methods(ans_corpus: List[str], 
                      raw_ans_corpus: List[str], 
                      quest_corpus: List[str], 
                      raw_quest_corpus: List[str]):

    """
    большая функция, индексирующая корпуса всеми моделями сразу (в мейне этого скрипта не вызывается)
    """
    
    count_vectorizer, ans_count = index_count(ans_corpus, corpus_type='answer')
    _, quest_count = index_count(quest_corpus, count_vectorizer=count_vectorizer, corpus_type='question')
     
    tfidf_vectorizer, ans_tfidf = index_tfidf(ans_corpus, corpus_type='answer')
    _, quest_tfidf = index_tfidf(quest_corpus, tf_idf_vectorizer=tfidf_vectorizer, corpus_type='question')
    
    ans_bm25 = index_bm25(ans_corpus)
    
    ans_fasttext = index_fasttext(ans_corpus, corpus_type='answer')
    quest_fasttext = index_fasttext(quest_corpus, corpus_type='question')
    
    ans_bert = index_bert(raw_ans_corpus, corpus_type='answer')
    quest_bert = index_bert(raw_quest_corpus, corpus_type='question')
    
    return ans_count, quest_count, \
           ans_tfidf, quest_tfidf, \
           ans_bm25, ans_fasttext, quest_fasttext, \
           ans_bert, quest_bert
    

def get_similarity_scores(query: str, indexed_data: np.array, vect_type='fasttext') -> np.array:
    """
    функция подсчета близости query и индексированного корпуса для fasttext и bert
    """
    if vect_type == 'fasttext':
        query_vect = get_text_vector(query)
    else:
        query_vect = embed_bert_cls(query, model, tokenizer)
        
    scores = cosine_similarity(indexed_data, query_vect.reshape(1, -1))

    return scores


def get_sorted_results(query: str,
           indexed_data: np.array,
           vect_type: str,
           ans_corpus: np.array) -> np.array:
    """
    полная функция запроса: принимает запрос, выдает отсортированные по релевантности выдачи из корпуса
    """
    scores = get_similarity_scores(query, indexed_data, vect_type)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    answer = ans_corpus[sorted_scores_indx.ravel()]  

    return answer


def main():
    processed_data, raw_data = get_corpus('questions_about_love.jsonl')
    
    ans_corpus, quest_corpus = processed_data[0], processed_data[1]
    raw_ans_corpus, raw_quest_corpus = raw_data[0], raw_data[1]
    
    quest_fasttext = index_fasttext(quest_corpus, corpus_type='question')
    quest_bert = index_bert(raw_quest_corpus, corpus_type='question')    
    
    do_search = 1

    while do_search:
        query = input('Введите ваш запрос: ')
        index_type = input('Введите название модели, которую нужно использовать для поиска (fasttext или bert): ')
        
        indexed_data = quest_fasttext if index_type == 'fasttext' else quest_bert
        if index_type == 'fasttext':
            query = preprocess(query)
        answer = get_sorted_results(query, indexed_data, index_type, np.array(raw_ans_corpus))
        print('Первые 5 ответов по релевантности: ', '\n -', '\n - '.join(answer[:5]))

        do_search = int(input('Введите 0, если хотите прекратить поиски, введите 1, если хотите сделать еще запрос: '))


if __name__ == '__main__':
    main()
