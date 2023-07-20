from langchain.document_loaders import TextLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import string

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', "data")


def get_example_txt_file_path():
    file_name = os.listdir(DATA_DIR)[0]
    file_dir = os.path.join(DATA_DIR, file_name)
    assert os.path.exists(file_dir)
    return file_dir


def clean_text(text: str) -> str:
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


def clean_doc(documents):
    documents_clean = []
    for d in documents:
        documents_clean.append(clean_text(d))
    return documents_clean


def load_example_data(file_path: str = get_example_txt_file_path()):
    documents = TextLoader(file_path, encoding='utf8').load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,)
    documents = splitter.split_documents(documents)
    documents = [doc.page_content for doc in documents]
    documents = clean_doc([str(doc).lower() for doc in documents])
    return documents
