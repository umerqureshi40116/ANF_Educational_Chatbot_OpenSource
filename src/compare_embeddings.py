############################GROQ#####################################

from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

emb1 = embedding_fn.embed_query("apple")
emb2 = embedding_fn.embed_query("iphone")

score = cosine_similarity([emb1], [emb2])[0][0]
print("Score:", score)

