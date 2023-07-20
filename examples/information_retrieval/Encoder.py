import math
import numpy as np
from typing import List
import copy


class TFIDFTransformer(object):
    def __init__(self,
                 document_dict: dict[int, str],
                 vocabularies: dict[str, int],
                 term_frequency_dict: dict[int, dict[str, int]],
                 inverse_document_frequency: dict[str, int]) -> None:
        """
        A class for fitting the given documents.
        Also allow you to search through the documents
        Using `search` function.
        Args:
            document_dict (dict[int, str]): A list of documents with indices
            vocabularies (dict[str, int]): A counter of vocabularies
                initialized as all zeros
            term_frequency_dict (dict[int, dict[str, int]]): tf
            inverse_document_frequency (dict[str, int]): idf
        """
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
        """
        class method for initialization from documents
        Args:
            documents (List[str]): corpus

        Returns:
            object: an object of TFIDFTransformer
        """
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
        """
        calculate `tf_word * idf` and collect them in a feature vector
        Args:
            term_frequency (dict[str, int]): tf
            inverse_document_frequency (dict[str, int]): idf

        Returns:
            np.ndarray: feature vector
        """

        tfidf = {word: tf_word * inverse_document_frequency[word]
                 for word, tf_word in term_frequency.items()}
        tfidf_vector = np.array([tfidf_word
                                 for _, tfidf_word in tfidf.items()])
        return tfidf_vector

    def transform(self, text: str) -> np.ndarray:
        """
        transform a cleaned text into a feature vector
        based on learned idf.
        Args:
            text (str): random clean query

        Returns:
            np.ndarray: feature vector
        """
        tf = self.compute_term_frequency(
            text=text,
            vocabularies=self.vocabularies
        )
        return self.calculate_feature_vector(
            term_frequency=tf,
            inverse_document_frequency=self.idf
        )

    def search(self, text: str, topk=5) -> list:
        """
        Similarity search through corpus given by a query
        Args:
            text (str): arbitrary clean query
            topk (int, optional): top k documents to return. Defaults to 5.

        Returns:
            list: _description_
        """
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
        """
        calculate cosine similarity
        Args:
            ref (np.ndarray): reference
            que (np.ndarray): query

        Returns:
            float: cosine similarity
        """
        return np.dot(ref, que)/(np.linalg.norm(ref)*np.linalg.norm(que))

    @classmethod
    def compute_term_frequency(cls,
                               text: str,
                               vocabularies: dict[str, int]) -> dict:
        """
        calculate term frequency
        Args:
            text (str): input text
            vocabularies (dict[str, int]): vocabulary list from corpus

        Returns:
            dict: a dict containing the tf for each word
        """
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
        """
        calculate the idf
        Args:
            documents (List[str]): a list of documents

        Returns:
            dict[str, float]: idf
        """
        # Total number of all documents
        N = len(documents)
        idf_dict = {}

        for document in documents:
            for word in set(document.split(' ')):
                # Count how many documents appear this word
                idf_dict[word] = idf_dict.get(word, 0) + 1

        # Apply logarithmic function to the counts
        idf_dict = {word: math.log(N / count)
                    for word, count in idf_dict.items()}
        # Consider unknown words in the testing
        # If we regard the query as an additional document
        # The this unknown word only appear in the query document
        # And the total number of documents should increase by 1
        # This is just a toy attempt for solving the unk word in testing
        idf_dict['[UNK]'] = math.log(N + 1 / 1)

        return idf_dict

    @classmethod
    def get_vocabularies(cls, documents: List[str]) -> dict:
        """
        get a vocabularies given by corpus
        Args:
            documents (List[str]): corpus

        Returns:
            dict: an empty counter
        """
        words = []
        for doc in documents:
            words.extend(doc.split(' '))
        words = set(words)
        words.add("[UNK]")  # consider unknown words in testing
        vocabularies = dict.fromkeys(words, 0)
        return vocabularies
