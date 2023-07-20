from Loader import clean_text, load_example_data
from Encoder import TFIDFTransformer

documents = load_example_data()

tfidf = TFIDFTransformer.fit(documents)
searching_query = "what does Russia’s Vladimir Putin seek to do?"

print(f"Searching query: {searching_query}")
clean_q = clean_text(searching_query)
print(f"Query after processing: {clean_q}")
results = tfidf.search(clean_q)[0]
print(f"Searched result: {results}")
