from sentence_transformers import SentenceTransformer, util

# Load Model
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "What is overfitting in machine learning?",
    "Explain the concept of overfitting.",
    "What is your favorite color?"
]

# Every sentence convert into embedding
embeddings = model.encode(sentences)

# Similarity calculation
similarities = util.cos_sim(embeddings[0], embeddings)

print("Comparing with:", sentences[0])
for i, sentence in enumerate(sentences):
    print(f"  '{sentence}' -> similarity: {similarities[0][i]:.4f}")