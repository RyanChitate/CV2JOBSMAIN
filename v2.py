import streamlit as st
import spacy
from PyPDF2 import PdfReader
from io import BytesIO
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Sample job descriptions (converted to lowercase)
job_descriptions = [
    "we are looking for a software engineer with experience in python.",
    '''Data Analyst – Excel expert

    The ideal candidate is adept at using large data sets to find insights and opportunities to enable business growth for our clients. We are looking for a detail-oriented, problem-solver who has proven capability to deliver business insights drawn from data.

    Qualifications & Experience

    At least three years in a support function in research, working intensively with Excel spreadsheets.
    At least an undergraduate degree in accounting, statistics, econometrics or other quantitative field.
    Outstanding Excel skills from basic functions through to entry-level macro programming.
    Able to produce graphs, tables, and other visual representations of data in an insightful and meaningful way.
    Strong problem-solving skills
    Excellent written and verbal communication skills for coordinating across teams.
    A drive to learn and master new technologies and techniques.''',
    "seeking a marketing manager with social media expertise."
]

# Function to calculate similarity between two texts using cosine similarity
def calculate_similarity(text1, text2):
    vec1 = nlp(text1).vector.reshape(1, -1)
    vec2 = nlp(text2).vector.reshape(1, -1)
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return similarity * 100

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Preprocess and clean text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

# Rank jobs by CV similarity
def rank_jobs(uploaded_cvs, uploaded_filenames):
    ranked_jobs = []
    for cv, filename in zip(uploaded_cvs, uploaded_filenames):
        job_scores = []
        cv_cleaned = preprocess_text(cv)
        for job_desc in job_descriptions:
            job_cleaned = preprocess_text(job_desc)
            score = calculate_similarity(cv_cleaned, job_cleaned)
            job_scores.append(score)
        ranked_jobs.append({"cv_filename": filename, "job_scores": job_scores})
    return ranked_jobs

# Main app
def main():
    plt.style.use('dark_background')

    st.title("🧭 CareerCompass")
    st.write("## ✍️ CV Ranking System")
    st.markdown("---")

    with st.sidebar:
        st.title("🧭 CareerCompass")
        st.write("Welcome to CareerCompass, your personal career guide!")
        st.write("We help you find the most suitable job opportunities based on the similarity between your CV and job descriptions.")
        st.markdown("---")
        st.markdown("# 💀 Cheat Code")
        if st.button("💡 Show Team Members"):
            st.markdown("👤 Mitheel Ramdaw")
            st.markdown("👤 Ryan Chitate")
            st.markdown("👤 Mikhaar Ramdaw")
            st.markdown("👤 Ashley Chitate")
        st.markdown("---")

    st.header("📄 Upload Your CVs")
    uploaded_files = st.file_uploader("Upload CV PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Rank Jobs") and uploaded_files:
        uploaded_cvs = []
        uploaded_filenames = []
        for uploaded_file in uploaded_files:
            cv_text = extract_text_from_pdf(uploaded_file)
            uploaded_cvs.append(cv_text)
            uploaded_filenames.append(uploaded_file.name)

        ranked_jobs = rank_jobs(uploaded_cvs, uploaded_filenames)

        for ranked_job in ranked_jobs:
            with st.container(border=True):
                st.subheader(f"📄 CV: {ranked_job['cv_filename'].split('.pdf')[0]}")
                job_scores = ranked_job['job_scores']

                sorted_scores = sorted(
                    zip(job_descriptions, job_scores),
                    key=lambda x: x[1],
                    reverse=True
                )

                for j, (job_desc, score) in enumerate(sorted_scores):
                    short_desc = job_desc.strip().split('\n')[0][:80]
                    st.markdown(f"**Job {j + 1}:** _{short_desc}..._\n🔗 Similarity Score: **{score:.2f}%**")

                # Plotting
                plt.figure(figsize=(10, 4))
                sns.barplot(x=list(range(1, len(job_scores) + 1)), y=job_scores)
                plt.title("Job Similarity Scores", color='white')
                plt.xlabel("Job", color='white')
                plt.ylabel("Score (%)", color='white')
                st.pyplot(plt)

                # Download CSV option
                df = pd.DataFrame({
                    "Job Description": [f"Job {i+1}" for i in range(len(job_scores))],
                    "Similarity (%)": [f"{s:.2f}" for s in job_scores]
                })
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download This CV's Results as CSV", csv, f"{ranked_job['cv_filename']}_ranking.csv", "text/csv")

if __name__ == "__main__":
    main()
