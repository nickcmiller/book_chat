import streamlit as st
import logging
import time

from query_functions import search_vector_db, query_data, get_unique_values, filter_json_by_key_value
from genai_toolbox.helper_functions.string_helpers import retrieve_file
from typing import List, Dict, Any

def select_books(book_index: Dict[str, Any]) -> None:
    if 'selected_books' not in st.session_state:
        st.session_state.selected_books = []
    
    # Let Streamlit handle the state update
    st.multiselect('Select Book:', book_index['books'], key='selected_books')
    print("selected_books: ", st.session_state.selected_books)

def get_chapters_for_books(book_index: Dict[str, Any], selected_books: List[str]) -> List[str]:
    all_chapters = []
    for book in selected_books:
        all_chapters.extend(book_index['chapters'][book])
    return all_chapters

def select_chapters(all_chapters: List[str]) -> None:
    if 'selected_chapters' not in st.session_state:
        st.session_state.selected_chapters = []
    
    # Filter out any previously selected chapters that are no longer available
    st.session_state.selected_chapters = [
        chapter for chapter in st.session_state.selected_chapters
        if chapter in all_chapters
    ]
    
    # Let Streamlit handle the state update
    st.multiselect(
        'Select Chapter:', 
        all_chapters, 
        default=st.session_state.selected_chapters,
        key='selected_chapters'
    )
    print("selected_chapters: ", st.session_state.selected_chapters)

def select_books_and_chapters(book_index: Dict[str, Any]) -> None:
    col1, col2 = st.columns(2)
    
    with col1:
        select_books(book_index)
    
    all_chapters = get_chapters_for_books(book_index, st.session_state.selected_books)
    
    with col2:
        select_chapters(all_chapters)
    
    if not st.session_state.selected_chapters and st.session_state.selected_books:
        st.warning("No specific chapters selected. All chapters from the selected books will be available for searching.")

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
    source: dict,
    not_summary: bool = True,
    index: int = None
) -> None:
    """
    Displays the source information in the Streamlit sidebar.

    Parameters:
    - source (dict): A dictionary containing source information.
    - summary (bool): A boolean indicating whether the source is a summary or not.
    """

    top_text = f"""
    **Author:** {source['author']}

    **Chapter:** {source['chapter']}

    """
    if not_summary:
        top_text += f"**Text:**\n"
    else:
        top_text += f"**AI Generated Summary:**\n"

    with st.sidebar:
        st.header(f"{index + 1}: {source['title']}")
        st.markdown(top_text)
        with st.expander(label="Expand Text", expanded=not_summary):
            st.markdown(source['text'])

def handle_user_input(
    prompt: str,
    chat_history: list[dict],
    filtered_chapters: List[Dict[str, Any]]
) -> None:
    book_list= retrieve_file("./extracted_documents/all_books_paragraphs.json" )

    st.sidebar.empty() 
    st.sidebar.title("Sources")
   
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching for sources..."):
        # print(f"filtered_chapters: {len(filtered_chapters)}")
        similar_chunks = search_vector_db(
            question=prompt,
            dict_list=filtered_chapters,
            history_messages=chat_history,
        )

        summary_sources = []
        for i, source in enumerate(similar_chunks):
            if source['type'] == "paragraph":
                display_source_info(source, index=i)
            else:
                display_source_info(source, not_summary=False, index=i)

    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
               
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
    st.set_page_config(
        page_title="Book Chat",
        page_icon=":book:",
        initial_sidebar_state="expanded"
    )

    book_index = retrieve_file("./extracted_documents/book_and_chapter_index.json")

    # Initialize session state for selected chapters if it doesn't exist
    if 'selected_chapters' not in st.session_state:
        st.session_state.selected_chapters = []

    # Call select_books_and_chapters and update session state
    new_selected_chapters = select_books_and_chapters(book_index)
    if new_selected_chapters:
        st.session_state.selected_chapters = new_selected_chapters

    print("selected_chapters: ", st.session_state.selected_chapters)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "How may I help?"}]

    display_chat_history(st.session_state.chat_history)

    prompt = st.chat_input("What would you like to know?")

    if prompt:
        handle_user_input(
            prompt=prompt, 
            chat_history=st.session_state.chat_history, 
            filtered_chapters=filtered_chapters,
        )

if __name__ == "__main__":
    main()