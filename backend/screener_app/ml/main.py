from docx import Document
import io
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from screener_app.ml.preprocessing import clean_text
from screener_app.ml.matcher import match_resume_to_jd
from screener_app.ml.semantic_similarity import semantic_similarity


app = FastAPI()

@app.post("/match")
async def match_resume(job_description: str, file: UploadFile = File(...)):
    
    contents = await file.read()
    doc = Document(io.BytesIO(contents))

    raw_text = "\n".join([para.text for para in doc.paragraphs])


    resume_text = clean_text(raw_text)  # your function
    score = match_resume_to_jd(resume_text, job_description) # your function
    semantic_score = semantic_similarity(resume_text,job_description) # semantic score
    
    
    
    return {"semanticScore": semantic_score}
