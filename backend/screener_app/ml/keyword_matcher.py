import re

# ---------------------------------------------
# 🧩 Step 1: Define Predefined Skill Weights
# ---------------------------------------------
# You can later automate these weights based on JD analysis.
skill_weights = {
    "python": 0.9,
    "machine learning": 1.0,
    "deep learning": 0.85,
    "data science": 0.95,
    "django": 0.7,
    "flask": 0.6,
    "data analysis": 0.75,
    "nlp": 0.8,
    "sql": 0.7,
    "pandas": 0.65,
    "numpy": 0.65,
    "tensorflow": 0.9,
    "pytorch": 0.9,
    "api": 0.6,
    "git": 0.5,
    "communication": 0.4
}

# ---------------------------------------------
# ⚙️ Step 2: Extract Skills from Text
# ---------------------------------------------
def extract_skills(text):
    """
    Extract skills mentioned in the text based on the skill_weights dictionary.
    Uses case-insensitive matching.
    """
    text = text.lower()
    found_skills = [skill for skill in skill_weights.keys() if re.search(r'\b' + re.escape(skill) + r'\b', text)]
    return found_skills


# ---------------------------------------------
# 📊 Step 3: Weighted Skill Matching Function
# ---------------------------------------------
def weighted_skill_match(resume_text, jd_text):
    """
    Calculate weighted skill match score between resume and job description.
    The score depends on how many weighted skills from JD are found in resume.
    """
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    
    if not jd_skills:
        return 0.0  # no skill keywords found in JD

    total_weight = sum(skill_weights[skill] for skill in jd_skills)
    matched_weight = sum(skill_weights[skill] for skill in resume_skills if skill in jd_skills)
    
    weighted_score = matched_weight / total_weight if total_weight > 0 else 0.0
    return round(weighted_score, 2)


# ---------------------------------------------
# 🧪 Step 4: Example Usage
# ---------------------------------------------
if __name__ == "__main__":
    resume_text = "Python developer experienced in ML, NLP, and TensorFlow"
    jd_text = "Looking for a Machine Learning Engineer skilled in Python, NLP, and Django"
    
    score = weighted_skill_match(resume_text, jd_text)
    
    print("Resume Skills:", extract_skills(resume_text))
    print("JD Skills:", extract_skills(jd_text))
    print("Weighted Skill Match Score:", score)
