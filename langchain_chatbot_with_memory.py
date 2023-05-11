# Install packages
!pip -q install openai langchain huggingface_hub transformers

# Import packages
import os
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory, ConversationKGMemory
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your OPENAI APIkey'

# Initialize OpenAI with the Davinci-003 model
llm = OpenAI(model_name='text-davinci-003', temperature=0, max_tokens=256)

# Define the template for the knowledge graph conversation
template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# Define different memory types for the conversation chain
buffer_memory = ConversationBufferMemory()
summary_memory = ConversationSummaryMemory(llm=OpenAI())
window_memory = ConversationBufferWindowMemory(k=2)
summary_buffer_memory = ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40)
kg_memory = ConversationKGMemory(llm=llm)

# Create conversation chains with different memory types
conversation_with_buffer_memory = ConversationChain(llm=llm, verbose=True, memory=buffer_memory)
conversation_with_summary_memory = ConversationChain(llm=llm, verbose=True, memory=summary_memory)
conversation_with_window_memory = ConversationChain(llm=llm, verbose=True, memory=window_memory)
conversation_with_summary_buffer_memory = ConversationChain(llm=llm, memory=summary_buffer_memory, verbose=True)
conversation_with_kg_memory = ConversationChain(llm=llm, verbose=True, prompt=prompt, memory=kg_memory)

# Start conversations with each conversation chain
conversation_with_buffer_memory.predict(input="Hi there! I am Sam")
conversation_with_summary_memory.predict(input="Hi there! I am Sam")
conversation_with_window_memory.predict(input="Hi there! I am Sam")
conversation_with_summary_buffer_memory.predict(input="Hi there! I am Sam")
conversation_with_kg_memory.predict(input="Hi there! I am Sam")
