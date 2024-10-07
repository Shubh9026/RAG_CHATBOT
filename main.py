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

tools = [determine_intent, get_user_age]    

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

def get_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_storage

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
        vector_embeddings = get_embeddings(text_chunks)

        thread_id = str(uuid.uuid4())

        # Store embeddings and config in memory
        conversation = {"vector_embeddings": vector_embeddings, "thread_id": thread_id}
        
        return {"message": "PDFs uploaded and processed successfully!", "data": {"thread_id": thread_id}}
    raise HTTPException(status_code=400, detail="No files uploaded")

# Define the request model for questions
class QuestionRequest(BaseModel):
    question: str
    thread_id: str

# Existing ask endpoint for RAG
@app_rag.post("/ask/")
async def ask_question(question_request: QuestionRequest):
    try:
        print(f"Received request: {question_request}")
        
        if conversation is None or conversation["thread_id"] != question_request.thread_id:
            raise HTTPException(status_code=400, detail="Invalid thread ID or no PDF processed.")
            
        vector_embeddings = conversation["vector_embeddings"]
        response = vector_embeddings.similarity_search(question_request.question, k=1)
        
        if response:
            answer = response[0].page_content
        else:
            answer = "Sorry, I couldn't find an answer."
            
        return {"response": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Step 5: Create a FastAPI endpoint to interact with the agent
@app_agent.post("/ask-agent/")
async def ask_agent_question(question_request: QuestionRequest):
    global conversation  # Ensure conversation is accessible
    try:
        print(f"Received query: {question_request}")
        # Check if conversation exists and the thread ID is valid
        if conversation is None or conversation["thread_id"] != question_request.thread_id:
            raise HTTPException(status_code=400, detail="Invalid thread ID or no PDF processed.")

        # Use the agent to determine the intent of the question
        inputs = {"messages": [("user", question_request.question)]}
        intent = determine_intent.invoke(question_request.question)
        
        print(f"received intent:{intent}")

        # Handle PDF-related queries by searching the vector store
        if intent == "general_query":
            print(1)
            vector_embeddings = conversation["vector_embeddings"]  # Fetch stored PDF embeddings
            response = vector_embeddings.similarity_search(question_request.question, k=1)  # Perform similarity search
            print(f"ram:{response}")
            if response:
                return {"response": f"PDF-related answer: {response[0].page_content}"}
            else:
                return {"response": "No relevant PDF content found for your query."}

        # If the intent is a greeting, respond with a greeting message
        elif intent == "greeting":
            return {"response": "Hello! How can I assist you today?"}

        # Otherwise, handle general queries
        else:
            print(2)
            inputs = {"messages": [("user", question_request.question)]}
            response = agent.run(inputs, config={"configurable": {"thread_id": question_request.thread_id}})
            
            # If agent response is valid, return the last message
            if "messages" in response and len(response["messages"]) > 0:
                return {"response": response["messages"][-1][1]}  # Extract response from agent's output
            else:
                raise HTTPException(status_code=500, detail="Agent did not return a valid response.") # Extract response from agent's output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Main execution block to run both applications
if __name__ == "__main__":
    import uvicorn
    import threading

    # Function to run the RAG app
    def run_rag_app():
        uvicorn.run(app_rag, host="0.0.0.0", port=8000)

    # Function to run the agent app
    def run_agent_app():
        uvicorn.run(app_agent, host="0.0.0.0", port=8001)

    # Start both applications in separate threads
    threading.Thread(target=run_rag_app).start()
    threading.Thread(target=run_agent_app).start()
