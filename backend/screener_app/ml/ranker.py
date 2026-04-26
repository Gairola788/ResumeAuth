# ranker.py
from keyword_matcher import weighted_skill_match
from semantic_similarity import semantic_similarity
from matcher import match_resume_to_jd

def rank_resumes(job_description: str, resumes: list[str]) -> list[dict]:
    """
    Rank resumes against a job description using 3 scoring methods:
    - Semantic similarity (BERT) → 50% weight
    - TF-IDF cosine similarity   → 30% weight
    - Weighted keyword matching  → 20% weight
    """
    
    # Get TF-IDF scores for all resumes at once
    tfidf_results = match_resume_to_jd(resumes, job_description)
    tfidf_scores = {i: score for i, score in tfidf_results}

    results = []

    for i, resume_text in enumerate(resumes):
        # Individual scores
        bert_score    = semantic_similarity(resume_text, job_description)
        tfidf_score   = tfidf_scores.get(i, 0.0)
        keyword_score = weighted_skill_match(resume_text, job_description)

        # Weighted final score
        final_score = (
            bert_score    * 0.5 +
            tfidf_score   * 0.3 +
            keyword_score * 0.2
        )

        results.append({
            "resume_index"  : i + 1,
            "bert_score"    : round(bert_score, 3),
            "tfidf_score"   : round(tfidf_score, 3),
            "keyword_score" : round(keyword_score, 3),
            "final_score"   : round(final_score, 3)
        })

    # Sort by final score — highest first
    results.sort(key=lambda x: x["final_score"], reverse=True)

    return results


# Test it
if __name__ == "__main__":
    jd = "Looking for a Machine Learning Engineer skilled in Python, NLP, TensorFlow and Django"
    
    resumes = [
        "Python developer with experience in NLP and TensorFlow",
        "Java developer with Spring Boot and SQL experience",
        "Data Scientist skilled in Python, Machine Learning and Pandas"
    ]

    rankings = rank_resumes(jd, resumes)

    for rank, result in enumerate(rankings, 1):
        print(f"\nRank #{rank}")
        print(f"  Resume Index  : {result['resume_index']}")
        print(f"  BERT Score    : {result['bert_score']}")
        print(f"  TF-IDF Score  : {result['tfidf_score']}")
        print(f"  Keyword Score : {result['keyword_score']}")
        print(f"  Final Score   : {result['final_score']}")