import streamlit as st
import logging
import time

from query_functions import search_vector_db, query_data



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

def display_source_info(
    source: dict
) -> None:
    """
    Displays the source information in the Streamlit sidebar.

    Parameters:
    - source (dict): A dictionary containing source information.
    """
    st.sidebar.header(source['title'])

    top_text = f"""
    **Author:** {source['author']}

    **Chapter:** {source['chapter']}

    **Text:**\n
    """
    st.sidebar.markdown(top_text)
    with st.sidebar.expander("View Source Text"):
        st.sidebar.markdown(source['text'])

def handle_user_input(
    prompt: str,
    chat_history: list[dict],
    file_path: str
) -> None:

    st.sidebar.empty() 
   

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        similar_chunks = search_vector_db(
            question=prompt,
            file_path=file_path,
            history_messages=chat_history,
        )

        for source in similar_chunks:
            time.sleep(0.1)  # Pause for 100 milliseconds
            display_source_info(source)
        
        query_answer = query_data(
                question=prompt,
                similar_chunks=similar_chunks,
                history_messages=chat_history, 
        )
            
        for chunk in query_answer:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
        chat_history.extend([
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": full_response}
        ])

# Main application setup
def main():
    file_path = "./extracted_documents/all_books_paragraphs.json"   

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
        handle_user_input(
            prompt, 
            st.session_state.chat_history, 
            file_path
        )

if __name__ == "__main__":
    main()