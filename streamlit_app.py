import streamlit as st
import logging

from query_functions import query_data

file_path = "./extracted_documents/all_books_paragraphs.json"

def display_chat_history(
    chat_history: list[dict]
) -> None:
    """
        Displays the chat history in the Streamlit application.

        Parameters:
        - chat_history (list[dict]): A list of dictionaries representing the chat history, 
        where each dictionary contains the role (user or assistant) and the content of the message.

        This function iterates through the chat history and displays each message in the chat interface.
        Messages are displayed according to their role, using the appropriate Streamlit chat message format.
    """
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(
    prompt: str,
    chat_history: list[dict]
) -> None:
    """
        Handles user input by processing the prompt and updating the chat history.

        Parameters:
        - prompt (str): The user's input message to be processed.
        - chat_history (list[dict]): A list of dictionaries representing the chat history, 
        where each dictionary contains the role (user or assistant) and the content of the message.

        This function performs the following steps:
        1. Appends the user's prompt to the chat history.
        2. Logs the current chat history for debugging purposes.
        3. Displays the user's message in the chat interface.
        4. Queries the OpenAI API for a response based on the user's prompt and the chat history.
        5. Displays the assistant's response in the chat interface.
        6. Appends the assistant's response to the chat history.

        Returns:
        - None: This function does not return any value but updates the chat history and the chat interface.
    """
    chat_history.append({"role": "user", "content": prompt})
    logging.info(f"Chat history: {chat_history}")

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = query_data(file_path, chat_history, prompt)  
        message_placeholder.markdown(full_response)

        chat_history.append({"role": "assistant", "content": full_response})

st.title("Simple Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

display_chat_history(st.session_state.chat_history)

prompt = st.chat_input("What is up?")

if prompt:
    handle_user_input(prompt, st.session_state.chat_history)