import chromadb
import json

# ChromaDB client — persistent storage (save on disk)
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# Create Collection  (if already exist then use that collection also)
collection = chroma_client.get_or_create_collection(name="ds_interview_questions")


def load_questions_into_chromadb():
    """Question bank ko ek baar ChromaDB mein load karta hai."""
    with open("data/questions.json", "r") as f:
        questions = json.load(f)

    # If Collection already filled then don't overload
    if collection.count() > 0:
        print("Questions already loaded in ChromaDB.")
        return

    ids = [str(q["id"]) for q in questions]
    documents = [q["question"] for q in questions]
    metadatas = [{"topic": q["topic"], "level": q["level"]} for q in questions]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Loaded {len(questions)} questions into ChromaDB.")


def get_relevant_questions(query: str, n_results: int = 3):
    """Given a topic/query, sabse relevant questions retrieve karta hai."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results["documents"][0]  # list of matching question texts