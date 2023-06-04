
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load PDF documents
def load_documents(file_path):
    loader = UnstructuredPDFLoader(file_path)
    return loader.load()

# Split documents into chunks
def chunk_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(data)

# Answer questions using document search and retrieval chain
def answer_questions(query, docsearch, chain):
    results = []
    results.append(query)
    docs = docsearch.similarity_search(query, include_metadata=True)
    ans = chain.run(input_documents=docs, question=query)
    results.append(ans["answer"])
    return results

# Load PDF file
file_name = 'Robotic Summer Camp Flyer.pdf'
data = load_documents(file_name)


# Chunk documents
texts = chunk_documents(data)
print(f'Now you have {len(texts)} documents')

# Set up embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma.from_documents(texts, embedding_function="gpt-turbo-3.5")

# Set up retrieval QA chain
llm = ChatOpenAI(temperature=0.0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "document_separator": "<<<<>>>>>"
    }
)

# hard coded answers
examples = [
    {
        "query": "what can we get from the camp?",
        "answer": "build, program, and drive their own \
            robots, compete in minigames"
    },
    {
        "query": "What age range is the Robotics Summer Camp geared towards?",
        "answer": "The Robotics Summer Camp is geared towards rising 3rd-7th graders."
    }
]

# Answer questions
result = qa.run(examples[0]["query"])


result = qa.run(examples[1]["query"])


# QA Generation
from langchain.evaluation.qa import QAGenerateChain
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI())

new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data]
)

examples += new_examples
examples

# Predict answers
predictions = qa.apply(examples)
predictions

# Evaluate the answers
from langchain.evaluation.qa import QAEvalChain
llm = ChatOpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)
graded_outputs

