
import sys
import pinecone
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

def load_documents(file_path):
    loader = UnstructuredPDFLoader(file_path)
    return loader.load()

def chunk_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(data)

def create_pinecone_index(texts, embeddings):
    index_name = "langchain2"
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
    return Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

def answer_questions(query, docsearch, chain):
    results=[]
    results.append(query)
    docs = docsearch.similarity_search(query, include_metadata=True)
    ans = chain.run(input_documents=docs, question=query)
    results.append(ans)
    return results



if __name__ == "__main__":
    print(sys.path)

    OPENAI_API_KEY = 'your openAI API key'
    PINECONE_API_KEY = 'your pinecone API key'
    PINECONE_API_ENV = 'asia-northeast1-gcp'


    data = load_documents("./data/Chip Huyen - Designing Machine Learning Systems_ An Iterative Process for Production-Ready Applications-O'Reilly Media (2022).pdf")
    print(f'You have {len(data)} document(s) in your data')
    print(f'There are {len(data[0].page_content)} characters in your document')
    # %%
    texts = chunk_documents(data)
    print(f'Now you have {len(texts)} documents')

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    docsearch = create_pinecone_index(texts, embeddings)

    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")

    results = []

    query = "What is the transfer learning?"
    results += answer_questions(query, docsearch, chain)

    query = "can you provide page number of model monitoring?"
    results += answer_questions(query, docsearch, chain)

    query = "summarize different data sampling strategy?"
    results += answer_questions(query, docsearch, chain)

    query = "quote the key summary about model drifting and how to fix it?"
    results += answer_questions(query, docsearch, chain)

    query = "which chapter and page number author talks about model drifting?"
    results += answer_questions(query, docsearch, chain)

    query = "what is the author's name and where did he or she go to school?"
    results += answer_questions(query, docsearch, chain)

    with open('output.txt', 'w') as f:
        for item in results:
                f.write("%s\n" % item)

# %%
