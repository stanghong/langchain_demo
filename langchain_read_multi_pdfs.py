import sys, os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')

pdf_folder_path = f'./pdfs'
os.listdir(pdf_folder_path)

# point 1 load multiple pdfs
# Vector Store
# Chroma as vectorstore to index and search embeddings
# There are three main steps going on after the documents are loaded:
# Splitting documents into chunks
# Creating embeddings for each document
# Storing documents and embeddings in a vectorstore
# location of the pdf file/files. 
loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)  if fn not in ['.DS_Store']]
# %%
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, 
             temperature=0,
             model_name="gpt-3.5-turbo")
# %%
# creat embedding once and save as vectorstore
index = VectorstoreIndexCreator().from_loaders(loaders)

# %% include the llm to avoid I do not know
index.query_with_sources('ask me three questions relating to these pdfs?', llm)
# %%
index.query_with_sources('summarize the transformer steps?')
# %%
index.query_with_sources('compare encoder and decoder functions in these pdfs and explain the differences?')
