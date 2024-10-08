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
from redis import Redis
import json
import os



# Connect to Redis
redis_client = Redis(host='localhost', port=6379, decode_responses=True)

# Directory to store FAISS indices
FAISS_INDEX_DIR = "./faiss_indices"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# Replace conversation global with Redis functions
def set_conversation(thread_id, index_path):
    conversation_data = {"thread_id": thread_id, "index_path": index_path}
    redis_client.set("conversation", json.dumps(conversation_data))

def get_conversation():
    conversation_data = redis_client.get("conversation")
    if conversation_data:
        return json.loads(conversation_data)
    return None

# Load environment variables
load_dotenv()

# FastAPI app for RAG
app_rag = FastAPI()

# FastAPI app for the agent
app_agent = FastAPI()

# In-memory storage for chat state and embeddings
conversation = None

# Initialize the memory saver
memory_saver = MemorySaver()

# Use the OpenAI model
model = ChatOpenAI(model="gpt-4")

# Define the tools for the agent
@tool("Determine the intent of the user's question")
def determine_intent(question: str) -> str:
    """Determine if the user's question is a greeting, PDF-related, or a general query."""
    if "hi" in question.lower() or "hello" in question.lower():
        return "greeting"
    elif any(keyword in question.lower() for keyword in ["pdf", "document", "content"]):
        return "pdf_related"
    else:
        return "general_query"

@tool("get_user_age")
def get_user_age(name: str) -> str:
    """Use this tool to find the user's age, currently hardcoded for specific names."""
    if "bob" in name.lower():
        return "42 years old"
    return "41 years old"

tools = [get_user_age]    

# Create the agent with memory to track conversation state
agent = create_react_agent(model, tools=tools, checkpointer=MemorySaver())

# PDF processing functions
def get_pdf_content(documents):
    raw_text = ""
    for document in documents:
        pdf_reader = PdfReader(io.BytesIO(document))
        for page in pdf_reader.pages:
            raw_text += page.extract_text() or ""
    return raw_text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_embeddings(chunks, thread_id):
    embeddings = OpenAIEmbeddings()
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    # Save the FAISS index to a file
    index_path = os.path.join(FAISS_INDEX_DIR, f"{thread_id}.index")
    vector_storage.save_local(index_path)
    
    return index_path

# Define a POST endpoint for PDF uploads in the RAG app
@app_rag.post("/upload/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    global conversation
    if files:
        extracted_text = ""
        for file in files:
            contents = await file.read()
            extracted_text += get_pdf_content([contents])

        text_chunks = get_chunks(extracted_text)
        thread_id = str(uuid.uuid4())

        # Get embeddings and save FAISS index path
        index_path = get_embeddings(text_chunks, thread_id)
        
        # Store conversation in Redis
        set_conversation(thread_id, index_path)
        return {"message": "PDFs uploaded and processed successfully!", "data": {"thread_id": thread_id}}
    raise HTTPException(status_code=400, detail="No files uploaded")

# Define the request model for questions
class QuestionRequest(BaseModel):
    question: str
    thread_id: str

# Existing ask endpoint for RAG
@app_rag.post("/ask/")
async def ask_question(question_request: QuestionRequest):
    print(1)
    
    conversation = get_conversation()
    if not conversation or conversation["thread_id"] != question_request.thread_id:
        raise HTTPException(status_code=400, detail="Invalid thread ID or no PDF processed.")
    print(2)
    index_path = conversation["index_path"]
    print(3)
    # Load FAISS index from file
    vector_storage = FAISS.load_local(index_path, OpenAIEmbeddings(), allow_dangerous_deserialization = True)
    response = vector_storage.similarity_search(question_request.question, k=1)
    print(4)
    
    answer = response[0].page_content if response else "Sorry, I couldn't find an answer."
    return {"response": answer}
        
    

# Endpoint to interact with the agent
@app_agent.post("/ask-agent/")
async def ask_agent_question(question_request: QuestionRequest):
    try:
        conversation = get_conversation()
        if not conversation or conversation["thread_id"] != question_request.thread_id:
            raise HTTPException(status_code=400, detail="Invalid thread ID or no PDF processed.")
        print(1)
        intent = determine_intent.invoke(question_request.question)
        print(intent)
        if intent == "pdf_related":
            index_path = conversation["index_path"]
            vector_storage = FAISS.load_local(index_path, OpenAIEmbeddings())
            response = vector_storage.similarity_search(question_request.question, k=1)
            return {"response": f"PDF-related answer: {response[0].page_content}" if response else "No relevant PDF content found for your query."}

        if intent == "greeting":
            return {"response": "Hello! How can I assist you today?"}
        print(2)
        if intent == "general_query":
            print(3)
            messages = [HumanMessage(content=question_request.question)]
            response = agent.invoke({"messages": messages, "input": question_request.question}, config={"configurable": {"thread_id": question_request.thread_id}})
            print(response)
            print(4)
            ai_contents = [message.content for message in response['messages'] if 'AIMessage' in str(type(message))]

            return {"response": ai_contents[0]} if ai_contents else HTTPException(status_code=500, detail="No AIMessage content found in the agent's response.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main execution block to run both applications
if __name__ == "__main__":
    import uvicorn
    from multiprocessing import Process

    def run_rag_app():
        uvicorn.run(app_rag, host="0.0.0.0", port=8000, log_level="info")

    def run_agent_app():
        uvicorn.run(app_agent, host="0.0.0.0", port=8001, log_level="info")

    process_rag = Process(target=run_rag_app)
    process_agent = Process(target=run_agent_app)
    process_rag.start()
    process_agent.start()
    process_rag.join()
    process_agent.join()