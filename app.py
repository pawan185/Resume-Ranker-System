
import streamlit as st
import pandas as pd
import os
import tempfile
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Ranking System", layout="wide")

st.title("📄 SelectMatrix: Precision in Candidate Shortlisting")

# User-defined weight ratios
st.sidebar.header("Adjust Score Weights")
similarity_weight = st.sidebar.slider("Similarity Score Weight (%)", 0, 100, 20)
skill_weight = st.sidebar.slider("Skill Match Score Weight (%)", 0, 100, 50)
experience_weight = st.sidebar.slider("Experience Match Score Weight (%)", 0, 100, 30)

total_weight = similarity_weight + skill_weight + experience_weight
if total_weight != 100:
    st.sidebar.warning("⚠️ Total weight must be 100%! Adjust the sliders accordingly.")

st.header("1️⃣ Upload Job Description")
jd_input = st.text_area("Paste Job Description here:", height=200)
jd_file = st.file_uploader("Or upload Job Description file (.txt)", type=['txt'])

if jd_file is not None:
    jd_input = jd_file.read().decode('utf-8')

resume_files = st.file_uploader("Upload Multiple Resumes (PDF only)", type=["pdf"], accept_multiple_files=True)

TECHNICAL_SKILLS = {"python", "java", "c++", "tensorflow", "aws", "docker", "react", "nlp", "sql"}  # Sample

def extract_skills(text):
    words = re.findall(r'\b[a-zA-Z0-9+#.]+\b', text.lower())
    return list(set(words) & TECHNICAL_SKILLS)

def extract_experience(text):
    exp_patterns = [r'(\d+)\s*(?:years?|yrs?)', r'(\d+)\s*(?:months?)']
    total_experience = 0
    for pattern in exp_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            years = int(match)
            if "months" in pattern:
                years = years / 12
            total_experience += years
    return min(50, round(total_experience, 1))

if st.button("🚀 Rank Resumes"):
    if not jd_input.strip():
        st.warning("Please enter a job description before ranking resumes.")
    elif not resume_files:
        st.warning("Please upload at least one resume.")
    elif total_weight != 100:
        st.warning("Please ensure the total weight equals 100%.")
    else:
        with st.spinner("Processing resumes... Please wait..."):
            data = []
            jd_skills = extract_skills(jd_input)

            for resume_file in resume_files:
                tmp_path = os.path.join(tempfile.gettempdir(), resume_file.name)
                with open(tmp_path, 'wb') as out:
                    out.write(resume_file.getbuffer())
                
                with fitz.open(tmp_path) as doc:
                    resume_text = " ".join(page.get_text() for page in doc)
                os.remove(tmp_path)

                documents = [jd_input, resume_text]
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(documents)
                similarity_score = round(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100, 2)
                
                resume_skills = extract_skills(resume_text)
                matched_skills = list(set(jd_skills) & set(resume_skills))
                skill_match_score = round((len(matched_skills) / max(len(jd_skills), 1)) * 100, 2)
                
                experience = extract_experience(resume_text)
                experience_required = extract_experience(jd_input)
                experience_match_score = min(100, round((experience / max(experience_required, 1)) * 100, 2))
                
                final_score = round(((similarity_weight / 100) * similarity_score) + 
                                    ((skill_weight / 100) * skill_match_score) + 
                                    ((experience_weight / 100) * experience_match_score), 2)

                data.append({
                    'Candidate Name': resume_file.name.replace('.pdf', ''),
                    'Resume File': resume_file.name,
                    'Similarity Score (%)': similarity_score,
                    'Matched Skills': ', '.join(matched_skills) if matched_skills else "None",
                    'Skill Match Score (%)': skill_match_score,
                    'Experience (Years)': experience,
                    'Experience Match Score (%)': experience_match_score,
                    'Final Score (Out of 100)': final_score
                })

            df = pd.DataFrame(data).sort_values(by='Final Score (Out of 100)', ascending=False).reset_index(drop=True)
            df.index = df.index + 1
            st.success("✅ Resumes ranked successfully!")
            st.dataframe(df)

            @st.cache_data
            def convert_df_to_excel(dataframe):
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    dataframe.to_excel(writer, index=False, sheet_name='Ranked Resumes')
                return output.getvalue()

            st.download_button("📥 Download Results as Excel", data=convert_df_to_excel(df), file_name="ranked_resumes.xlsx")
