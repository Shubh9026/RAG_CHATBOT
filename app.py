


import streamlit as st
import requests
from typing import Dict, Optional

# Initialize session state variables
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def upload_files(files) -> bool:
    """Handle file upload and set thread_id"""
    try:
        files_to_upload = [("files", (file.name, file, "application/pdf")) for file in files]
        response = requests.post(
            "http://127.0.0.1:8000/upload/",
            files=files_to_upload,
            timeout=60
        )
        
        if response.ok:
            data = response.json()
            thread_id = data.get("data", {}).get("thread_id")  # Ensure we're accessing the thread ID correctly
            if thread_id:
                st.session_state.thread_id = thread_id
                print(f"Thread ID set to: {thread_id}")  # Debug print
                return True
            else:
                st.error("No thread ID received from server")
                return False
        else:
            st.error(f"Upload failed: {response.status_code}")
            return False
            
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return False



def ask_rag_question(question: str, thread_id: str) -> Optional[Dict]:
    """Send question to RAG backend"""
    if not thread_id:
        st.error("No thread ID available. Please upload a PDF first.")
        return None
        
    try:
        payload = {
            "question": question,
            "thread_id": thread_id
        }
        st.info(f"Sending payload to RAG: {payload}")  # Debug print
        
        response = requests.post(
            "http://127.0.0.1:8000/ask/",  # Point to RAG app
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.ok:
            return response.json()
        else:
            error_detail = response.json().get("detail", "Unknown error")
            st.error(f"Error: {error_detail}")
            return None
            
    except Exception as e:
        st.error(f"Request error: {str(e)}")
        return None

##THIS IS ASLO ADDED FOR THE AGENT PORT 8001

def ask_agent_question(question: str, thread_id: str) -> Optional[Dict]:
    """Send question to Agent backend"""
    if not thread_id:
        st.error("No thread ID available. Please upload a PDF first.")
        return None
        
    try:
        payload = {
            "question": question,
            "thread_id": thread_id
        }
        st.info(f"Sending payload to Agent: {payload}")  # Debug print
        
        response = requests.post(
            "http://127.0.0.1:8001/ask-agent/",  # Point to Agent app
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.ok:
            return response.json()
        else:
            error_detail = response.json().get("detail", "Unknown error")
            st.error(f"Error: {error_detail}")
            return None
            
    except Exception as e:
        st.error(f"Request error: {str(e)}")
        return None        


def main():
    st.title("Chat with PDFs - Baasha, the PDF ChatBot")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner('Processing PDFs...'):
            if upload_files(uploaded_files):
                st.success("PDFs processed successfully!")
                st.session_state.chat_history = []  # Clear chat history on new upload
    
    # Display thread ID status (for debugging)
    st.sidebar.write(f"Current Thread ID: {st.session_state.thread_id}")
    
    ## Create tabs for RAG and Agent, NEW ONE
    tab1, tab2 = st.tabs(["RAG Queries", "Agent Queries"])
     
    ##LOGIC FOR THE TAB APPEARANCE
    with tab1:
        st.subheader("Ask a RAG Question")
        if st.session_state.thread_id:
            user_query = st.text_input("How can I help you today?", key="rag_user_query")
            if st.button("Ask RAG"):
                if user_query:
                    response = ask_rag_question(user_query, st.session_state.thread_id)

                    if response:
                        bot_response = response.get("response", "I don't have an answer for that.")
                        st.session_state.chat_history.append({"user": user_query, "bot": bot_response})

                        # Display the chat history
                        st.markdown("### Chat History")
                        for chat in st.session_state.chat_history:
                            st.markdown(f"**You**: {chat['user']}")
                            st.markdown(f"**Baasha**: {chat['bot']}")
        else:
            st.error("Please upload and process a PDF first.")

    with tab2:
        st.subheader("Ask an Agent Question")
        if st.session_state.thread_id:
            user_query = st.text_input("How can I help you with the Agent?", key="agent_user_query")
            if st.button("Ask Agent"):
                if user_query:
                    response = ask_agent_question(user_query, st.session_state.thread_id)

                    if response:
                        bot_response = response.get("response", "I don't have an answer for that.")
                        st.session_state.chat_history.append({"user": user_query, "bot": bot_response})

                        # Display the chat history
                        st.markdown("### Chat History")
                        for chat in st.session_state.chat_history:
                            st.markdown(f"**You**: {chat['user']}")
                            st.markdown(f"**Baasha**: {chat['bot']}")
        else:
            st.error("Please upload and process a PDF first.") 

if __name__ == "__main__":
    main()
