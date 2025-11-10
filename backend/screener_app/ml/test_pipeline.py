from screener_app.ml.semantic_similarity import semantic_similarity
from screener_app.ml.preprocessing import clean_text
from screener_app.ml.vectorizer import ResumeVectorizer
from screener_app.ml.matcher import match_resume_to_jd
from screener_app.ml.keyword_matcher import weighted_skill_match

# 🧾 Sample Resumes
resumes = [
    "Experienced Data Scientist with skills in Python, Machine Learning, and NLP.",
    "Software Engineer skilled in Java, Spring Boot, and Microservices.",
    "AI/ML Engineer with hands-on in TensorFlow, PyTorch, and Scikit-learn."
]

# 🧠 Job Description
jd = "Looking for a Python developer with ML and NLP experience."

print("------ Cleaned Resumes ------")
for r in resumes:
    print(clean_text(r))


vec = ResumeVectorizer()
X = vec.fit_transform(resumes)

print("\n---- TF-IDF Shape ----")
print(X.shape)

print("\n---- Example Feature Vector ----")
print(X[1])  # Resume 2

unique_words_count = (X[1] > 0).sum()
print("Unique words in Resume 2:", unique_words_count)

# Step: Evaluate each resume against JD
print("\n---- Resume vs JD Matching ----")
final_results = []

for i, resume in enumerate(resumes, 1):
    tfidf_score = match_resume_to_jd([resume], jd)[0][1]
    semantic_score = semantic_similarity(resume, jd)
    weighted_skill_score = weighted_skill_match(resume, jd)

    # Weighted combination (tuneable)
    final_score = (0.5 * semantic_score) + (0.3 * tfidf_score) + (0.2 * weighted_skill_score)


    final_results.append((i, final_score))
    print(f"Resume {i} -> TF-IDF: {tfidf_score:.2f}, Semantic: {semantic_score:.2f}, Weighted: {weighted_skill_score:.2f}, Final: {final_score:.2f}")

# Step 4️: Rank resumes
final_results.sort(key=lambda x: x[1], reverse=True)

print("\n🏆 Ranked Resumes (Best to Worst):")
for idx, score in final_results:
    print(f"Resume {idx}: Final Combined Score = {score:.2f}")
