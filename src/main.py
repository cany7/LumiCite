from ingestion.JSON_builder import build_json
from indexing.vector_store import *
from pathlib import Path
# build_json()

FAISS_PATH = Path("data/my_faiss.index")
TEXT_PATH = Path("data/text_data.pkl")

searcher = FaissSearch(index_path=FAISS_PATH, text_data_path=TEXT_PATH)
searcher.load_index()
question = "Whats your favorite color?"  # <--- Change your query here
results = searcher.search(question, k=3)

print(results[0]["text"] + "\n\n\n\n")
print(results[1]["text"] + "\n\n\n\n")
print(results[2]["text"] + "\n\n\n\n")