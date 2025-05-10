import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ========== SETTINGS ==========
st.set_page_config(page_title="CareerCompass - CV2Jobs", layout="wide")
plt.style.use('dark_background')

# ========== MODEL LOAD ==========
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_model()

# ========== TEXT UTILITIES ==========
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_similarity(text1, text2):
    vec1 = model.encode([text1])
    vec2 = model.encode([text2])
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return similarity

# ========== JOB MATCHING LOGIC ==========
def rank_jobs(uploaded_cvs, uploaded_filenames, job_descriptions):
    ranked_jobs = []
    for cv, filename in zip(uploaded_cvs, uploaded_filenames):
        job_scores = []
        cv_cleaned = preprocess_text(cv)
        for job_desc in job_descriptions:
            job_cleaned = preprocess_text(job_desc)
            score = calculate_similarity(cv_cleaned, job_cleaned)
            job_scores.append(score)

        # 🔁 Normalise so best job = 100%
        max_score = max(job_scores)
        if max_score > 0:
            job_scores = [(s / max_score) * 100 for s in job_scores]
        else:
            job_scores = [0 for _ in job_scores]

        ranked_jobs.append({"cv_filename": filename, "job_scores": job_scores})
    return ranked_jobs

# ========== STREAMLIT APP ==========
def main():
    st.title("🧭 CareerCompass – CV2Jobs @ Delta0")
    st.write("### Internal CV-to-Job Ranking Tool")
    st.markdown("---")

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Delta_Logo.svg/1280px-Delta_Logo.svg.png", width=150)
        st.subheader("Team Only")
        st.write("This tool is for Delta0 internal use only.")
        if st.button("💡 Show Team Members", key="show_team"):
            st.markdown("👤 Mitheel Ramdaw")
            st.markdown("👤 Ryan Chitate")
            st.markdown("👤 Mikhaar Ramdaw")
            st.markdown("👤 Ashley Chitate")
        st.markdown("---")
        uploaded_job_file = st.file_uploader("📄 Upload Job Descriptions CSV", type=["csv"], key="job_uploader")
        job_descriptions = []
        if uploaded_job_file:
            job_df = pd.read_csv(uploaded_job_file)
            if "description" in job_df.columns:
                job_descriptions = job_df["description"].dropna().tolist()
                st.success(f"Loaded {len(job_descriptions)} job descriptions.")
            else:
                st.error("CSV must contain a 'description' column.")

    st.header("📄 Upload Your CVs")
    uploaded_files = st.file_uploader("Upload PDF CVs", type=["pdf"], accept_multiple_files=True, key="cv_uploader")

    run_ranking = st.button("🚀 Rank Jobs", key="run_ranking")

    if run_ranking and uploaded_files and job_descriptions:
        uploaded_cvs = []
        uploaded_filenames = []
        for uploaded_file in uploaded_files:
            cv_text = extract_text_from_pdf(uploaded_file)
            uploaded_cvs.append(cv_text)
            uploaded_filenames.append(uploaded_file.name)

        ranked_jobs = rank_jobs(uploaded_cvs, uploaded_filenames, job_descriptions)

        for ranked_job in ranked_jobs:
            with st.container(border=True):
                st.subheader(f"📄 CV: {ranked_job['cv_filename'].split('.pdf')[0]}")
                job_scores = ranked_job['job_scores']

                # Sort scores and jobs together
                sorted_scores = sorted(
                    zip(job_descriptions, job_scores),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Text output
                for j, (job_desc, score) in enumerate(sorted_scores):
                    short_desc = job_desc.strip().split('\n')[0][:80]
                    st.markdown(f"**Job {j + 1}:** _{short_desc}..._\n🔗 Relative Fit Score: **{score:.2f}%**")

                # Chart output (aligned with sorted order)
                sorted_labels = [f"Job {j+1}" for j, _ in enumerate(sorted_scores)]
                sorted_values = [score for _, score in sorted_scores]

                plt.figure(figsize=(10, 4))
                sns.barplot(x=sorted_labels, y=sorted_values, palette="Blues_d")
                plt.title("Relative Fit Scores (Best Match = 100%)", color='white')
                plt.xlabel("Job", color='white')
                plt.ylabel("Relative Score (%)", color='white')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(plt)

                # ✅ CSV aligned to sorted output
                df = pd.DataFrame({
                    "Job Description": sorted_labels,
                    "Relative Fit (%)": [f"{s:.2f}" for s in sorted_values]
                })
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download This CV's Results as CSV", csv, f"{ranked_job['cv_filename']}_ranking.csv", "text/csv")

    elif run_ranking:
        st.warning("Please upload both CVs and job descriptions first.")

if __name__ == "__main__":
    main()
