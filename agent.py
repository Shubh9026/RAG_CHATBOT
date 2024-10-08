import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import io 



# Load environment variables
load_dotenv()

# FastAPI app for RAG
app_rag = FastAPI()

# FastAPI app for the agent
app_agent = FastAPI()

# In-memory storage for chat state and embeddings
conversation = None

# Initialize the memory saver
memory = MemorySaver()

# Use the OpenAI model
model = ChatOpenAI(model="gpt-4")

# Define the tools for the agent
@tool
def determine_intent(question: str) -> str:
    """Determine the intent of the user's question."""
    if "hi" in question.lower() or "hello" in question.lower():
        return "greeting"
    elif any(keyword in question.lower() for keyword in ["pdf", "document", "content"]):
        return "pdf_related"
    else:
        return "general_query"

@tool
def get_user_age(name: str) -> str:
    """Use this tool to find the user's age."""
    if "bob" in name.lower():
        return "42 years old"
    return "41 years old"

tools = [get_user_age, determine_intent]

# Create the agent with memory to track conversation state
agent = create_react_agent(model,tools,checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "abc123"}}
content = "my name is shubh"
response = agent.invoke(
    {"messages": [HumanMessage(content=content)]}, config
)
print(response)
# Extract and print only the `content` of `AIMessage` entries
ai_contents = [message.content for message in response['messages'] if 'AIMessage' in str(type(message))]
print(ai_contents[0])

