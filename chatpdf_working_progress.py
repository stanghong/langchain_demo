# %%

# %%
import gradio as gr
import sys, os
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')

# %%
OPENAI_API_KEY
# %%
def load_documents(file_path):
    loader = UnstructuredPDFLoader(file_path)
    return loader.load()

def chunk_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(data)

def answer_questions(query, docsearch, chain):
    results=[]
    results.append(query)
    docs = docsearch.similarity_search(query, include_metadata=True)
    ans = chain.run(input_documents=docs, question=query)
    results.append(ans["answer"])
    return results


# def run_model(file, question):
    # Load PDF file
file_name='Robotic Summer Camp Flyer.pdf'
data = load_documents(file_name)
data
# %%
# data = load_documents(file.name)
    # print(f'You have {len(data)} document(s) in your data')
    # print(f'There are {len(data[0].page_content)} characters in your document')
    
    # Chunk documents
texts = chunk_documents(data)
print(f'Now you have {len(texts)} documents')
# %%
    # Set up embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma.from_documents(texts,  embedding_function="gpt-turbo-3.5") 
    # vectorstore = Chroma.from_documents(texts,  embeddings) # use more expensive model
    
    # Set up memory and conversational retrieval chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), 
    vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)
# %%
question='what is this document about?'
    # Answer question
result = qa({"question": question})
#     return result["answer"]
# %%
result
# %%
# QA Generation
from langchain.evaluation.qa import QAGenerateChain
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI())

# %%
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data]
)
# %%
new_examples
# %%
import langchain
langchain.debug = False
# %%
# predictions = qa.apply(new_examples[0]['query'])
question=new_examples[0]['query']
predictions =  qa({"question": question})
predictions
# %%
from langchain.evaluation.qa import QAEvalChain
llm = ChatOpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)
# %%
# # Create Gradio interface
# file_upload = gr.inputs.File(label="Upload PDF file")
# question = gr.inputs.Textbox(label="Question")
# output = gr.outputs.Textbox()

# gr.Interface(
#     fn=run_model,
#     inputs=[file_upload, question],
#     outputs=output,
#     title="Conversational Retrieval Chain",
#     description="Upload a PDF file and ask a question related to its content.",
#     # examples=[["./data/fulltext.pdf", "What is the paper about?"], ["./data/fulltext.pdf", "How is the cwsi defined?"]]
# ).launch(share=True)



# %%
