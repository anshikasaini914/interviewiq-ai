from app.services.vector_store import load_questions_into_chromadb, get_relevant_questions

# Step 1: Load Questions in ChromaDB  (First time needed only)
load_questions_into_chromadb()

# Step 2: Test after Query
query = "tell me about Python data structures"
results = get_relevant_questions(query, n_results=3)

print(f"\nQuery: {query}")
print("Top matching questions:")
for i, question in enumerate(results, 1):
    print(f"  {i}. {question}")