from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(text1, text2):
    emb = model.encode([text1, text2], convert_to_tensor=True)
    score = util.cos_sim(emb[0], emb[1])
    return score.item()
