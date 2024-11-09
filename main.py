import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text
import fitz  # PyMuPDF for reading PDF files

def extract_text_from_pdf(file):
    """Extract text from uploaded PDF file."""
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text

def create_streamlit_app(llm, portfolio, clean_text):
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")

    # Sidebar for user details
    st.sidebar.title("User Details")
    user_name = st.sidebar.text_input("Your Name")
    user_role = st.sidebar.text_input("Your Current Role")

    # Main page content
    st.title("📧 Job Application Mail Generator")
    url_input = st.text_input("Enter a Job URL:", value="https://boards.greenhouse.io/spacex/jobs/7560266002?gh_jid=7560266002")
    resume_upload = st.file_uploader("Upload your resume (PDF format):", type=["pdf"])
    submit_button = st.button("Generate Email")

    if submit_button:
        if resume_upload is not None and user_name and user_role:
            try:
                # Extract and clean resume text
                resume_text = extract_text_from_pdf(resume_upload)
                resume_text = clean_text(resume_text)
                proj = chain.protfolio_csv(resume_text)
                print(proj, "Hello")
                # Load job data from the provided URL
                loader = WebBaseLoader([url_input])
                job_data = clean_text(loader.load().pop().page_content)

                # Process the job and resume data
                portfolio.load_portfolio(proj)
                jobs = llm.extract_jobs(job_data)
                for job in jobs:
                    skills = job.get('skills', [])
                    links = portfolio.query_links(skills)

                    # Generate email with user details, job info, and resume text
                    email = llm.write_mail(job, links, resume=resume_text, name=user_name, role=user_role)
                    st.code(email, language='markdown')
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            if not resume_upload:
                st.warning("Please upload your resume to proceed.")
            if not user_name or not user_role:
                st.warning("Please complete the user details in the sidebar.")

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio, clean_text)
