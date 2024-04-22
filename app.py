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
from src.mcqgen.mcqgenerator import generate_evaluate_chain
import streamlit as st
from langchain.llms import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
import traceback

with open('response.json', 'r') as file:
    response_json=json.load(file)
 
st.title("MCQ Creator App with Langchain:")

with st.form("user_input"):
    uploaded_file=st.file_uploader("Upload a PDF or txt file")

    mcq_count=st.number_input("No. of MCQs", min_value=3, max_value=50)

    subject=st.text_input("Insert a subject", max_chars=20)

    tone=st.text_input("Difficulty level", max_chars=20, placeholder="Simple")

    button=st.form_submit_button("Create MCQs")

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text=read_file(uploaded_file)

                with get_openai_callback() as cb:   
                    response=generate_evaluate_chain (
                        {"text":text,
                        "number":mcq_count,
                        "subject":subject,
                        "tone":tone,
                        "response_json":json.dumps(response_json)}
                    )
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")

            else:
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost: {cb.total_cost}")
                if isinstance(response,dict):
                    quiz=response.get("quiz")
                    
                    if quiz is not None:
                        print('About to execute probalematic function')
                        table_data=get_table_data(quiz)
                        
                        
                        if table_data is not None:
                            df=pd.DataFrame(table_data)
                            df.index=df.index+1
                            st.table(df)
                            st.write(response.get("review"))
                        else:
                            st.error("Error in the table data")

                else:
                    st.write(response)
