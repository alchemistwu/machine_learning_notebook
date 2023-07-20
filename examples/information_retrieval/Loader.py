from langchain.document_loaders import TextLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import string
from typing import List
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', "data")


def get_example_txt_file_path() -> str:
    """
    return the path of an example txt file
    Returns:
        str: an absolute path to the example text file
    """
    file_name = os.listdir(DATA_DIR)[0]
    file_dir = os.path.join(DATA_DIR, file_name)
    assert os.path.exists(file_dir)
    return file_dir


def clean_text(text: str) -> str:
    """
    Toy method for cleaning text
    Args:
        text (str): dirty text

    Returns:
        str: cleaned text
    """
    # Remove Unicode
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove Mentions
    text = re.sub(r'@\w+', '', text)
    # Lowercase the document
    text = text.lower()
    # Remove punctuations
    text = re.sub(r'[%s]' % re.escape(string.punctuation),
                  ' ',
                  text)
    # Lowercase the numbers
    text = re.sub(r'[0-9]', '', text)
    # Remove the doubled space
    text = re.sub(r'\s{2,}', ' ', text)
    return text


def clean_doc(documents: List) -> List[str]:
    """
    Cleaning a list of docs.
    Args:
        documents (List): a list of documents

    Returns:
        List: a list of cleaned documents
    """
    documents_clean = []
    for d in documents:
        documents_clean.append(clean_text(d))
    return documents_clean


def load_example_data(file_path: str = get_example_txt_file_path()):
    """
    Loading example text data. Including the following workflow:
    1. load the text file.
    2. split the text file into multiple documents
    3. clean each document in the list
    4. return the list of cleaned documents
    Args:
        file_path (str, optional): _description_. Defaults to get_example_txt_file_path().

    Returns:
        _type_: _description_
    """
    documents = TextLoader(file_path, encoding='utf8').load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,)
    documents = splitter.split_documents(documents)
    documents = [doc.page_content for doc in documents]
    documents = clean_doc([str(doc).lower() for doc in documents])
    return documents
