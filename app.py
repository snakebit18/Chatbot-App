import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os

from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot"

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful massistant . Please  repsonse to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,llm):
    llm=ChatGroq(model_name=llm,groq_api_key=groq_api_key)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

## #Title of the app
st.title("Q&A Chatbot With OpenAI")


## Select the OpenAI model
llm=st.sidebar.selectbox("Select Open Source model",["Gemma2-9b-It","llama3-8b-8192","llama3-70b-8192"])



## MAin interface for user input
st.write("Goe ahead and ask any question")
user_input=st.text_input("You:")



if user_input :
    response=generate_response(user_input,llm)
    st.write(response)
else:
    st.write("Please provide the user input")


