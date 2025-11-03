from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

emb1 = model.encode("Experienced in Python and machine learning.")
emb2 = model.encode("Skilled in AI and data analysis using Python.")

similarity = util.cos_sim(emb1, emb2)
print(similarity)
