import streamlit as st
import time
import logging

from query_functions import query_data

file_path = "./extracted_documents/The_Philosophical_Baby_all_paragraphs.json"

def display_chat_history(
    chat_history: list[dict]
) -> None:
    """Display chat messages from history."""
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(
    prompt: str,
    chat_history: list[dict]
):
    """Handle user input and generate assistant response."""
    # Add user message to chat history
    chat_history.append({"role": "user", "content": prompt})
    logging.info(f"Chat history: {chat_history}")

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Simulate assistant response using OpenAI API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = query_data(file_path, chat_history, prompt)  
        message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        chat_history.append({"role": "assistant", "content": full_response})

st.title("Simple Chat")

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history
display_chat_history(st.session_state.chat_history)

# Add system instructions and history messages to the OpenAI query call
system_instructions = "Respond as if you were Ozzy Osbourne."  # Define system instructions

prompt = st.chat_input("What is up?")

if prompt:
    handle_user_input(prompt, st.session_state.chat_history)