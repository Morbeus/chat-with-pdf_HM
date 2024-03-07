from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from typing_extensions import Concatenate
from langchain.vectorstores import FAISS
from langchain.llms import OpenAIChat

from PyPDF2 import PdfReader
import streamlit as st
import json
import os


def main():
    st.title("PDF Text Search with OpenAI")
    
    # File Upload
    st.sidebar.title("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    # Query Input
    query = st.text_input("Enter your query:", "What is sin^2x + cos^2x?")

    if uploaded_file:
        # Load configuration
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']

        # Read PDF content
        reader = PdfReader(uploaded_file)
        raw_text = ''
        for page in reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Create embeddings and document search
        embeddings = OpenAIEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)

        # Load QA chain
        chain = load_qa_chain(OpenAIChat(), chain_type="stuff")

        # Search and display results
        # st.subheader("Search Results:")
        docs = document_search.similarity_search(query)
        # for doc in docs:
        #     st.write(doc)

        # Run QA chain
        st.subheader("Answer:")
        chain_output = chain.run(input_documents=docs, question=query)
        st.write(chain_output)

if __name__ == "__main__":
    main()
