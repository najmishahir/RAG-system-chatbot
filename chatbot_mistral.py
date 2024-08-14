# Using mistralai/Mistral-7B-Instruct-v0.2
# adapted from: https://python.langchain.com/docs/use_cases/question_answering/quickstart/

import os
import constants

import streamlit as st
from getpass import getpass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = constants.APIKEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = constants.HUGGINGFACE_TOKEN
os.environ["LANGCHAIN_API_KEY"] = constants.LANGCHAIN
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "final-year-proj-mistral"

# def load_pdf_data(directory="pdf/"):
#     loader = PyPDFLoader('pdf/COMP2121 AssessmentBrief.pdf')
#     return loader.load()
def load_pdf_data(directory="pdf/."):
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()

def preprocess_data(data):
    full_texts = '\n'.join([doc.page_content for doc in data])
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(full_texts)

    vector_store = Chroma.from_texts(texts, embedding=OpenAIEmbeddings(model="text-embedding-ada-002"))
    retriever = vector_store.as_retriever()

    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_chain(retriever, llm):
    prompt_template = """You are an intelligent assistant expertise in answering questions and is dedicated to provide detailed, accurate, and relevant 
            information about COMP2121 Data Mining and Text Analytics module to university students. Your knowledge is derived exclusively from a specific document/article designated as the official module content. 
            Your responses must:

            1. Remain strictly within the boundaries of the Data Mining and Text Analytics module. If a question falls outside of this module, politely inform the user with a standardized response: 
            "This question is beyond the scope of this module."

            2. Are based solely on the information contained within the provided document/article. Answer only the specific question posed and avoid including any irrelevant information.

            3. Are clear, concise and directly address the question posed. Avoid providing overly broad or generic information that might detract from the specific focus of the module.
            
            Answer Template:
            {context}

            Question: {question}

            Your task is to apply critical thinking to interpret the question's intent, and craft a response that is informative, precise, and wholly relevant to the query at hand.
            """
    
    custom_prompt = PromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def main():
    st.title('COMP2121 Data Mining')
    st.markdown("""Hi! I am a Teaching Assistant Chatbot for Data Mining & Text Analytics 
                and I am here to help you with any questions you have""")
    
    # Initialise chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if input_text := st.chat_input("How can I help?"):
        # Display the user input
        with st.chat_message("user"):
            st.markdown(input_text)
            st.session_state.chat_history.append({"role": "user", "content": input_text})

        # Display the assistant response    
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            data = load_pdf_data()
            retriever = preprocess_data(data)

            repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
            llm =  HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.2, token=constants.HUGGINGFACE_TOKEN)
            conv_chain = setup_rag_chain(retriever, llm) 
        
            response = conv_chain.invoke(input_text)
            message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
