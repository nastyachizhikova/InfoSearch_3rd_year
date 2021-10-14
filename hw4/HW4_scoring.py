from HW4 import get_corpus, index_all_methods
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm


def get_similarity_matrix(answers_matrix, questions_matrix):
    """
    получаем матрицу схожести
    """
    similarity = answers_matrix.dot(questions_matrix.T)

    return similarity


def get_sorted_results(answers_matrix, questions_matrix):
    """
    сортируем результаты по схожести
    """
    scores = get_similarity_matrix(answers_matrix, questions_matrix)

    if isinstance(scores, csr_matrix):
        scores = scores.toarray()

    sorted_indx = np.argsort(scores, axis=0)[::-1, :]

    return sorted_indx


def count_precision(answers_matrix, questions_matrix, model_name: str):
    """
    считаем качество модели: берем отсортированную матрицу схожести, оставляем первые top-k,
    проверяем для top-k каждого запроса, есть ли в них правильный ответ
    """
    sorted_indx = get_sorted_results(answers_matrix, questions_matrix)
    k = 5
    top_k_answers = sorted_indx.T[:, :k]

    p_k = 0

    for indx, query_answers in tqdm(enumerate(top_k_answers), total=sorted_indx.shape[0],
                                    desc=f'Counting precision for {model_name} model'):
        if indx in query_answers:
            p_k += 1

    return p_k / sorted_indx.shape[0]


def main():
    processed_data, raw_data = get_corpus('questions_about_love.jsonl')

    ans_corpus, quest_corpus = processed_data[0], processed_data[1]
    raw_ans_corpus, raw_quest_corpus = raw_data[0], raw_data[1]

    ans_count, quest_count, \
    ans_tfidf, quest_tfidf, \
    ans_bm25, ans_fasttext, \
    quest_fasttext, \
    ans_bert, quest_bert = index_all_methods(ans_corpus, raw_ans_corpus, quest_corpus, raw_quest_corpus)

    #  нормализация ненормализованных векторов для унификации процесса подсчета близости запросов и ответов
    ans_count_norm = normalize(ans_count)
    quest_count_norm = normalize(quest_count)
    ans_fasttext = normalize(ans_fasttext)
    quest_fasttext = normalize(quest_fasttext)
    ans_bert = normalize(ans_bert)
    quest_bert = normalize(quest_bert)

    print('Качество CountVectorizer: ', count_precision(ans_count_norm, quest_count_norm, model_name='Count'))
    print('Качество TfIdfVectorizer: ', count_precision(ans_tfidf, quest_tfidf, model_name='TfIdf'))
    print('Качество BM-25: ', count_precision(ans_bm25, quest_count, model_name='BM-25'))
    print('Качество fasttext: ', count_precision(ans_fasttext, quest_fasttext, model_name='fasttext'))
    print('Качество BERT: ', count_precision(ans_bert, quest_bert, model_name='BERT'))


if __name__ == '__main__':
    main()
