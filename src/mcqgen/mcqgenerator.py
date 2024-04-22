import openai
import json
import langchain
import pandas as pd
from openai import OpenAI
from langchain_openai import OpenAI, ChatOpenAI
import os
from dotenv import load_dotenv
from src.mcqgen.utils import read_file, get_table_data
from src.mcqgen.logger import logging

from langchain.llms import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback



load_dotenv()

key=os.getenv("OPENAI_API_KEY")


llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)



Template='''
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to create a quiz of {number} multiple 
choice questions for {subject} in {tone} tone. Make sure that the questions are not repeated and check all
question to be confirming the text as well. Make sure to format your responses like {response_json} and use
it as a guide     
'''

quiz_generation_prompt=PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "respone_json"],
    template=Template
)

quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)


Template2="""
You are an expert english grammarian and a writer. Given a multiple choice Quiz for {subject} students,
you need to evaluate  the complexity of the quiz and give a complete analysis of the quiz. Only use 50 words
at max. If the quiz is not at par the with the cognitive and the analytical ability of the student, update
the quiz questions and change the tone such that it perfectly fits the student's analytical ability.
Quiz_MCQ:
{quiz}
"""

quiz_evaluation_prompt=PromptTemplate(input_variables=['subject','quiz'], template=Template2)
review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

generate_evaluate_chain=SequentialChain(chains=[quiz_chain,review_chain], input_variables=["text", "number", "subject", "tone", "response_json"], output_variables=['quiz', 'review'], verbose=True)

# number =6
# subject="Machine Learning"
# tone="simple"

# with get_openai_callback() as cb:
#     response=generate_evaluate_chain(
#         {   
#             "text":text,
#             "number":number,
#             "subject":subject,
#             "tone":tone,
#             "response_json":json.dumps(response_json)
#         }
# )
    



