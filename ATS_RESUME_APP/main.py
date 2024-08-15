import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_text

# Function to clean and tokenize text
def clean_and_tokenize(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# Function to extract skills from resume
def extract_skills(text):
    skills = ['python', 'java', 'sql', 'data analysis', 'machine learning', 'project management']
    tokens = clean_and_tokenize(text)
    extracted_skills = [skill for skill in skills if skill in tokens]
    return extracted_skills

# Function to parse and extract text from PDF
def extract_text_from_pdf(file_path):
    return extract_text(file_path)

# Function to scan resume against job description
def scan_resume(resume_text, job_description):
    resume_tokens = clean_and_tokenize(resume_text)
    job_tokens = clean_and_tokenize(job_description)

    matching_words = set(resume_tokens).intersection(set(job_tokens))
    return matching_words

# Example usage
if __name__ == "__main__":
    # Sample job description
    job_description = """
    We are looking for a software engineer with experience in Python, Java, and SQL. 
    Knowledge in data analysis and machine learning is a plus.
    """

    # Path to resume PDF
    resume_path = 'sample_resume.pdf'

    # Extract text from resume
    resume_text = extract_text_from_pdf(resume_path)

    # Extract skills
    skills = extract_skills(resume_text)
    print(f"Extracted Skills: {skills}")

    # Scan resume against job description
    matching_words = scan_resume(resume_text, job_description)
    print(f"Matching Words: {matching_words}")
