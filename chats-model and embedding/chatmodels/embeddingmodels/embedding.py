from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# # =====================================================================
# #  Concept 1: Single Text (Query) ka Embedding nikalna -> embed_query
# # =====================================================================
# query_text = "What is Artificial Intelligence?"
# query_vector = embeddings.embed_query(query_text)

# print("=" * 60)
# print("1. SINGLE TEXT EMBEDDING (embed_query):")
# print(f"Text: '{query_text}'")
# print(f"Total Dimensions (Vector Length): {len(query_vector)}")
# print(f"Sample Vector Values (First 5 floats): {query_vector[:3072]}")
# print("=" * 60)


# =====================================================================
# 📌 Concept 2: Multiple Texts (Docs) ka Embedding nikalna -> embed_documents
# =====================================================================
documents = [
    "Machine learning enables computers to learn from data.",
    "Cricket is a popular sport played with a bat and ball.",
    "Deep neural networks are used in generative AI."
]

doc_vectors = embeddings.embed_documents(documents)

print("\n" + "=" * 60)
print("2. MULTIPLE DOCUMENTS EMBEDDING (embed_documents):")
for i, (doc, vec) in enumerate(zip(documents, doc_vectors), 1):
    print(f"Doc {i}: '{doc}'")
    print(f"  -> Dimensions: {len(vec)}, First 3 values: {vec[:3]}")
print("=" * 60)


# # =====================================================================
# # 📌 Concept 3: Semantic Similarity (Cosine Similarity) Example
# # Embeddings ka use karke pata lagana ki query kis document se sabse zyada match karti hai
# # =====================================================================
# def cosine_similarity(vec1, vec2):
#     dot_product = np.dot(vec1, vec2)
#     norm_vec1 = np.linalg.norm(vec1)
#     norm_vec2 = np.linalg.norm(vec2)
#     return dot_product / (norm_vec1 * norm_vec2)

# user_search = "How do computers learn automatically?"
# search_vector = embeddings.embed_query(user_search)

# print("\n" + "=" * 60)
# print(f"3. SIMILARITY SEARCH EXAMPLE:")
# print(f"Search Query: '{user_search}'\n")

# print("Similarity Scores (Highest score = Most relevant):")
# scores = []
# for doc, doc_vec in zip(documents, doc_vectors):
#     score = cosine_similarity(search_vector, doc_vec)
#     scores.append((doc, score))

# # Highest similarity score pehle display karenge
# scores.sort(key=lambda x: x[1], reverse=True)

# for doc, score in scores:
#     print(f"- Score: {score:.4f} | Document: '{doc}'")
# print("=" * 60)
