from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

# Example relation names
relations = ["works_at", "lives_in", "studied_at", "married_to", "employed_by"]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # or OpenAI embeddings

# Create embeddings
relation_embeddings = model.encode(
    relations, normalize_embeddings=True, convert_to_numpy=True
)

print(type(relation_embeddings))  # Should be <class 'numpy.ndarray'>
print(np.array(relation_embeddings).shape)  # Should be (num_vectors, dim)

# Build FAISS index
# data = np.random.rand(1000, 768)
index = faiss.IndexFlatIP(relation_embeddings.shape[1])
# index = faiss.IndexFlatL2(768)
print(index.is_trained)  # Should be True
index.add(relation_embeddings)

# When LLM gives a relation like "employed_by"
llm_output = "employed_by"
query_vec = model.encode([llm_output], normalize_embeddings=True, convert_to_numpy=True)

# Search
D, I = index.search(query_vec, k=1)
best_match = relations[I[0][0]]

print("LLM output:", llm_output)
print("Matched to:", best_match)
