# ResumeAuth

🚀 AI-powered Resume Screening Tool built with **Django + NLP**.  
Helps recruiters and candidates by analyzing resumes against job descriptions, detecting keyword stuffing, and providing smart scoring with BERT.

---

## Features
- 📂 Upload resumes (PDF/DOCX)
- 🔍 Extract and parse resume text
- 📊 Keyword matching with job description
- 🤖 AI Scoring (TF-IDF + BERT for semantic analysis)
- 🚫 Detect keyword stuffing
- 📑 Candidate & Recruiter Dashboards (future)

---

## Tech Stack
- Backend: Django, Django REST Framework
- AI/ML: Scikit-learn, HuggingFace Transformers
- Database: MySQL/Postgres
- Frontend: Django Templates / React (optional)
- Deployment: Render / Vercel / Heroku / AWS

---

## Setup Instructions
```bash
# Clone the repository
git clone https://github.com/your-username/ResumeAuth.git
cd ResumeAuth

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python manage.py runserver
