# Using GPT model
# adapted from: https://python.langchain.com/docs/use_cases/question_answering/quickstart/

import os
import constants
import streamlit as st

from operator import itemgetter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import AIMessage, get_buffer_string

os.environ["OPENAI_API_KEY"] = constants.APIKEY
os.environ["LANGCHAIN_API_KEY"] = constants.LANGCHAIN
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "final-year-proj"

def load_pdf_data(directory="pdf/."):
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def preprocess_data(data):
    full_texts = format_docs(data)
    
    # Indexing: Split (broken down into chunks)
    # Need to split the text such that it should not increase token size
    # overlap helps with mitigating the possiblity of separating words from important context related to it
    # RecursiveCharacterTextSplitter : recusrively splits the text into chunks of 1000 characters with 200 characters overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(full_texts)

    # Indexing: Store
    # Embed the text chunks and store them in a vector store
    # Perform a similarity search to identify the stored splits with most similar embeddings to our query
    # simplest measure of similarity is the cosine similarity
    vector_store = Chroma.from_texts(texts, embedding=OpenAIEmbeddings(model="text-embedding-ada-002"))

    # Retrieval and Generation: Retrieve (extracting relevant information from the pdf)
    # Retrieve and generate using the relevant documents (pdf)
    retriever = vector_store.as_retriever()

    return retriever


def setup_rag_chain(retriever, llm):
    contextualize_q_system_prompt = """Given the following conversation and a follow up question, 
            rephrase the follow up question to be a standalone question.

            Chat History: {chat_history}
            Follow Up Input: {question}
            Standalone question:
            """
    contextualize_q_prompt = PromptTemplate.from_template(contextualize_q_system_prompt)
    
    system_prompt = """You are an intelligent assistant expertise in answering questions and is dedicated to provide detailed, accurate, and relevant 
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

    prompt = PromptTemplate.from_template(system_prompt)
    
    inputs = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
        )
        | contextualize_q_prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    )
    rag_chain = {
        "context": itemgetter("standalone_question") | retriever | format_docs,
        "question": lambda x: x["standalone_question"],
    }
    conversation_rag_chain = inputs | rag_chain | prompt | llm 

    return conversation_rag_chain

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
        
            llm = ChatOpenAI(temperature=0.2, model_name="gpt-4-turbo")
            conv_chain = setup_rag_chain(retriever, llm)  
        
            response = conv_chain.invoke(
                {
                    "question": input_text, 
                    "chat_history": [AIMessage(content=st.session_state.chat_history),
                    ],
                }
            )
            message_placeholder.markdown(response.content + "| ")
            message_placeholder.markdown(response.content)
        st.session_state.chat_history.append({"role": "assistant", "content": response.content})      

if __name__ == "__main__":
    main()

