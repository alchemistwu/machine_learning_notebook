from examples.information_retrieval.Loader import clean_text, load_example_data
from examples.information_retrieval.Encoder import TFIDFTransformer

documents = load_example_data()

tfidf = TFIDFTransformer.fit(documents)

query = "what does Russia’s Vladimir Putin seek to do?"
clean_q = clean_text(query)

results = tfidf.search(clean_q)
for result in results:
    print(result)
    print("========================")