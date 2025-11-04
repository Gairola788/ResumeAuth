from screener_app.ml.semantic_similarity import semantic_similarity
from screener_app.ml.preprocessing import clean_text
from screener_app.ml.vectorizer import ResumeVectorizer
from screener_app.ml.matcher import match_resume_to_jd

#Sample resumes 
resumes = [
    "Experienced Data Scientist with skills in Python, Machine Learning, and NLP.",
    "Software Engineer skilled in Java, Spring Boot, and Microservices.",
    "AI/ML Engineer with hands-on in TensorFlow, PyTorch, and Scikit-learn."
]

#Step 1: Clean text
print("------Cleaned Resumes------")
for r in resumes:
     print(clean_text(r))
     
# Step 2: Vectorize
vec = ResumeVectorizer()
X = vec.fit_transform(resumes)

print("\n---- TF-IDF Shape ----")
print(X.shape)  # (3 resumes, 5000 features max)

print("\n---- Example Feature Vector ----")
print(X[1])  # Sparse vector representation

unique_words_count = (X[1] > 0).sum()
print("Unique words in Resume 2:", unique_words_count)



# Sample resumes
resumes1 = [
    "Experienced Data Scientist skilled in Python, Machine Learning, and NLP.",
    "Software Engineer skilled in Java, Spring Boot, and Microservices.",
    "AI/ML Engineer with hands-on TensorFlow, PyTorch, and Scikit-learn."
]

# Sample Job Description
jd = "Looking for a Python developer with ML and NLP experience."

print("---- Resume vs JD Matching ----")
results = match_resume_to_jd(resumes1, jd)

for idx, score in results:
    print(f"Resume {idx+1} -> Similarity: {score:.4f}")
    
    

resume_text = "Python developer with ML experience"
job_desc_text = "Experienced software engineer skilled in machine learning and Python"

semantic_score = semantic_similarity(resume_text, job_desc_text)
print(f"Semantic Match Score: {semantic_score:.2f}")

