# %%
import sys
print(sys.path)

# 
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 
loader = UnstructuredPDFLoader("./data/Chip Huyen - Designing Machine Learning Systems_ An Iterative Process for Production-Ready Applications-O'Reilly Media (2022).pdf")
data = loader.load()
# 
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')
# 
### Chunk your data up into smaller documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')
# 
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

OPENAI_API_KEY = 'your openAI API key'
PINECONE_API_KEY = 'your pinecone API key'
PINECONE_API_ENV = 'asia-northeast1-gcp'

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "langchain2"
# 
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
# 
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# 
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
# 
results=[]
query = "What is the transfer learning?"
docs = docsearch.similarity_search(query, include_metadata=True)
results.append(query)
# 
ans=chain.run(input_documents=docs, question=query)
results.append(ans)

# 


# 
query = "can you provide page number of model monitoring?"
docs = docsearch.similarity_search(query, include_metadata=True)
# 
ans= chain.run(input_documents=docs, question=query)
results.append(query)
results.append(ans)

# %%

query = "summarize different data sampling strategy?"
docs = docsearch.similarity_search(query, include_metadata=True)
# 
ans= chain.run(input_documents=docs, question=query)
results.append(query)
results.append(ans)

# %%

query = "quote the key summary about model drifting and how to fix it?"
docs = docsearch.similarity_search(query, include_metadata=True)
# 
ans= chain.run(input_documents=docs, question=query)
results.append(query)
results.append(ans)

# %%

query = "which chapter and page number author talks about model drifting?"
docs = docsearch.similarity_search(query, include_metadata=True)
# 
ans= chain.run(input_documents=docs, question=query)
results.append(query)
results.append(ans)

# %%

query = "what is the author's name and where did he or she go to school?"
docs = docsearch.similarity_search(query, include_metadata=True)
# 
ans= chain.run(input_documents=docs, question=query)
results.append(query)
results.append(ans)
# %%
with open('output.txt', 'w') as f:
    for item in results:
            f.write("%s\n" % item)


# %%
