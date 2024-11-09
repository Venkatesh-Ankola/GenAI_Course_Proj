import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]
    
    def protfolio_csv(self, resume_test):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM Resume:
            {resume}
            ### INSTRUCTION:
            The scraped text is from the resume of a user.
            Your job is to extract the projects, techstack and return them in csv format containing the following keys: `projects`, `teckstack`.
            Only return the valid CSV.
            ### VALID CSV (NO PREAMBLE):
            ### Example: 
            "Project","Techstack"
            "Amazon website","React, Node.js, MongoDB"
            "EngageX","Angular,.NET, SQL Server"
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"resume": resume_test})
        print(res.content,"Before")
        # try:
        #     json_parser = JsonOutputParser()
        #     res = json_parser.parse(res.content)
        #     print(res.content,"After")
        # except OutputParserException:
        #     raise OutputParserException("Context too big. Unable to parse jobs.")
        return res


    def write_mail(self, job, links, resume,name, role):
        prompt_email = PromptTemplate.from_template(
            # """
            # ### JOB DESCRIPTION:
            # {job_description}

            # ### INSTRUCTION:
            # You are Venkatesh Ankola, a student at RV University. AtliQ is an AI & Software Consulting company dedicated to facilitating
            # the seamless integration of business processes through automated tools. 
            # Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            # process optimization, cost reduction, and heightened overall efficiency. 
            # Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
            # in fulfilling their needs.
            # Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
            # Remember you are Mohan, BDE at AtliQ. 
            # Use the resume {resume}
            # Do not provide a preamble.
            # ### EMAIL (NO PREAMBLE):

            # """
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are {name}, with current role as {role}. You are trying to apply for a job. 
            Your job is to write an application email to the Hiring Manager regarding the job mentioned above describing the your capability. 
            Also add the most relevant ones from the following project to showcase your portfolio: {link_list}
            Remember you are Venkatesh Ankola, Student at RV University. 
            Use the resume {resume}. Select the best skills.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links, "resume": resume, "name":name, "role":role})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))