import os
import prompts
import openai
import streamlit as st

from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

# Load environment - loads the OpenAI API key
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

st.title('👨‍👩‍👧‍👦 👪  Nurture Chatbot 👨‍👨‍👦‍👦 👩‍👩‍👦‍👦')

def qa(file, query, chain_type, k):
    # load document
    loader = PyPDFLoader(file)
    documents = loader.load()
    
    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    print(result['result'])
    return result

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello ! My name is Nurture, How may i help you?"}
    ]

# Corrected usage of st.chat_message
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Correctly get user input and process it
if prompt := st.chat_input(placeholder="Ask me about parenthood ..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"):
        response = openai.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                        {"role": "system", "content": prompts.prompt1}
                    ] + [
                        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                    ],
            stream=True,
            # Include other necessary parameters for the API call
        )
    response_text = st.write_stream(response)
    st.session_state["messages"].append({"role": "assistant", "content": response_text})
    
    # # Display the chatbot response
    # with st.chat_message("assistant"):
    #     st.write(response_text)













# import os
# import openai
# import prompts

# import streamlit as st

# # Load environment - loads the OpenAI API key
# from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv('OPENAI_API_KEY')
# openai.api_key = api_key


# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
#     ]

# for msg in st.session_state["messages"]:
#     st.chat_message(msg["role"])(msg["content"])
        

# if prompt := st.chat_input(placeholder="Ask me about parenthood ..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)


# with st.chat_message("assistant"):
#         stream = openai.chat.completions.create(
#             model=st.session_state["openai_model"],
#             messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
#             stream=True,
#         )
#         response = st.write_stream(stream)
# st.session_state.messages.append({"role": "assistant", "content": response})