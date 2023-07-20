import math
from collections import Counter
import numpy as np
from typing import List
import copy


class TFIDFTransformer(object):
    def __init__(self,
                 document_dict: dict[int, str],
                 vocabularies: dict[str, int],
                 term_frequency_dict: dict[int, dict[str, int]],
                 inverse_document_frequency: dict[str, int]) -> None:
        self.document_dict = document_dict
        self.vocabularies = vocabularies
        self.tf_dict = term_frequency_dict
        self.idf = inverse_document_frequency
        self.tfidf_dict = {}
        for i, tf in self.tf_dict.items():
            self.tfidf_dict[i] = self.calculate_feature_vector(
                term_frequency=tf,
                inverse_document_frequency=self.idf
            )

    @classmethod
    def fit(cls, documents: List[str]) -> object:
        document_dict = {i: doc for i, doc in enumerate(documents)}
        vocabularies = cls.get_vocabularies(documents)
        tf_dict = {}
        for i, doc in document_dict.items():
            tf = cls.compute_term_frequency(
                text=doc,
                vocabularies=vocabularies
            )
            tf_dict[i] = tf
        idf = cls.compute_inverse_document_frequency(
            documents=documents
        )
        return cls(
            document_dict=document_dict,
            vocabularies=vocabularies,
            term_frequency_dict=tf_dict,
            inverse_document_frequency=idf
                   )

    @classmethod
    def calculate_feature_vector(
            cls,
            term_frequency: dict[str, int],
            inverse_document_frequency: dict[str, int]) -> np.ndarray:

        tfidf = {word: tf_word * inverse_document_frequency[word]
                 for word, tf_word in term_frequency.items()}
        tfidf_vector = np.array([tfidf_word
                                 for _, tfidf_word in tfidf.items()])
        return tfidf_vector

    def transform(self, text: str) -> dict:
        tf = self.compute_term_frequency(
            text=text,
            vocabularies=self.vocabularies
        )
        return self.calculate_feature_vector(
            term_frequency=tf,
            inverse_document_frequency=self.idf
        )

    def search(self, text: str, topk=5) -> list:
        query_feature = self.transform(text)
        searched_result = {}
        for index, feature in self.tfidf_dict.items():
            searched_result[index] = self.cos(query_feature, feature)

        sorted_searched_results = sorted(searched_result.items(),
                                         key=lambda x: -x[1])
        sorted_searched_results = dict(sorted_searched_results)
        results = []
        for i in range(topk):
            results.append(
                self.document_dict[
                    list(sorted_searched_results.keys())[i]])
        return results

    @classmethod
    def cos(cls, ref: np.ndarray, que: np.ndarray) -> float:
        return np.dot(ref, que)/(np.linalg.norm(ref)*np.linalg.norm(que))

    @classmethod
    def compute_term_frequency(cls,
                               text: str,
                               vocabularies: dict[str, int]) -> dict:
        words = text.split(' ')
        word_count_norm = copy.deepcopy(vocabularies)
        for word in words:
            if word in word_count_norm.keys():
                word_count_norm[word] += 1
            else:
                word_count_norm["[UNK]"] += 1
        for word, count in word_count_norm.items():
            word_count_norm[word] = count / len(words)
        return word_count_norm

    @classmethod
    def compute_inverse_document_frequency(
            cls,
            documents: List[str]) -> dict[str, float]:
        # Count how many documents contain each word
        N = len(documents)
        idf_dict = {}

        for document in documents:
            for word in set(document.split(' ')):
                idf_dict[word] = idf_dict.get(word, 0) + 1

        # Apply logarithmic function to the counts
        idf_dict = {word: math.log(N / count)
                    for word, count in idf_dict.items()}
        idf_dict['[UNK]'] = math.log(N + 1 / 1)

        return idf_dict

    @classmethod
    def get_vocabularies(cls, documents: List[str]) -> dict:
        words = []
        for doc in documents:
            words.extend(doc.split(' '))
        words = set(words)
        words.add("[UNK]")
        vocabularies = dict.fromkeys(words, 0)
        return vocabularies


def compute_tf(text):
    # Count frequency of each word in a document
    # and divide by total number of words
    words = text.split(' ')
    word_count = Counter(words)
    tf = {word: count/len(words) for word, count in word_count.items()}
    return tf


def compute_idf(documents):
    # Count how many documents contain each word
    N = len(documents)
    idf_dict = {}

    for document in documents:
        for word in set(document.split(' ')):
            idf_dict[word] = idf_dict.get(word, 0) + 1

    # Apply logarithmic function to the counts
    idf_dict = {word: math.log(N / count) for word, count in idf_dict.items()}

    return idf_dict


def compute_tfidf(documents):
    tfidf_documents = []

    idf = compute_idf(documents)
    print(idf)

    for document in documents:
        tf = compute_tf(document)
        tfidf = {word: tf_word * idf[word] for word, tf_word in tf.items()}
        tfidf_documents.append(tfidf)

    return tfidf_documents
