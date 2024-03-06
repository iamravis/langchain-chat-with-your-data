import os
import prompts
import openai
import streamlit as st
import dill
import pickle
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma





from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key


def main():
    st.title('🦜🔗 Nurture Chatbot')
    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    st.write(f'{pdf}')

    if pdf is not None:
        
        loader = PyPDFLoader(pdf.name)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        # Generate a database filename based on the PDF file name
        db_name = pdf.name[:-4]  # Assuming the PDF file has a '.pdf' extension
        db_file_path = f'{db_name}.pkl'
        st.write(f'{db_file_path}')
        # Load the document content from the uploaded file
            # This requires you to either read the PDF directly from the UploadedFile object or save it temporarily
        
        
        if os.path.exists(db_file_path):
            st.write("Loading the database from the pickle file.")
            with open(f'{db_file_path}', 'rb') as f:
                db = pickle.load(f)
        else:
            st.write("Creating a new database and embedding.")
            
            embeddings = OpenAIEmbeddings()  # Assuming this is correctly instantiated
            
            db = Chroma.from_documents(texts, embeddings)  # Ensure this operation is serializable
            with open(f'{db_file_path}', 'wb') as f:
                dill.dump(db, f)
        
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            
            # embeddings = OpenAIEmbeddings()
            # retriever = Chroma.from_documents(texts, embeddings).as_retriever(search_type="similarity", search_kwargs={"k": 3})
            # Assuming `db` supports `as_retriever` method correctly after being loaded
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            
            qa = ConversationalRetrievalChain.from_chain_type(
                llm=OpenAI(),  # Assuming OpenAI() is correctly instantiated
                chain_type="refine",
                retriever=retriever,
                return_source_documents=True
            )
            result = qa({"query": query})
            
            st.write(result)  # Display the result

if __name__ == '__main__':
    main()