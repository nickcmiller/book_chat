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

def display_source_info(source: dict) -> None:
    """
    Displays the source information in the Streamlit sidebar.

    Parameters:
    - source (dict): A dictionary containing source information.
    """
    st.sidebar.header(source['title'])
    st.sidebar.write(f"""
    **Author:**\n {source['author']}\n
    **Chapter:**\n {source['chapter']}\n
    **Text:**\n {source['text']}
    """)

def handle_user_input(
    prompt: str,
    chat_history: list[dict]
) -> None:

    st.sidebar.empty() 
    chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        query_answer, similar_chunks = query_data(
                question=prompt,
                file_path=file_path, 
                history_messages=chat_history, 
            )
        for source in similar_chunks:
            display_source_info(source)
            
        for chunk in query_answer:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
        chat_history.append({"role": "assistant", "content": full_response})

        

st.set_page_config(
    page_title="Book Chat",
    page_icon=":book:",
    initial_sidebar_state="expanded"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "How may I help?"}]

display_chat_history(st.session_state.chat_history)

prompt = st.chat_input("What is up?")

if prompt:
    handle_user_input(prompt, st.session_state.chat_history)