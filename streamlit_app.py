import streamlit as st
import logging
import time

from query_functions import search_vector_db, query_data, filter_by_criteria
from genai_toolbox.helper_functions.string_helpers import retrieve_file
from typing import List, Dict, Any
import json

def select_books_and_chapters(
    path_to_book_index: str,
) -> None:
    if 'book_index' not in st.session_state:
        st.session_state.book_index = retrieve_file(path_to_book_index)

    col1, col2 = st.columns(2)
    
    with col1:
        _select_books(st.session_state.book_index)
    
    # Ensure that chapters are updated when books are selected
    if 'selected_books' in st.session_state:
        all_chapters = _get_chapters_for_books(
            st.session_state.book_index, 
            st.session_state.selected_books
        )
    else:
        all_chapters = []

    with col2:
        _select_chapters(all_chapters)

def _select_books(
    book_index: Dict[str, Any]
) -> None:
    if 'selected_books' not in st.session_state:
        st.session_state.selected_books = []
    
    selected_books = st.multiselect(
        'Select Book:', 
        book_index['books'], 
        key='book_selector'
    )
    st.session_state.selected_books = selected_books

def _select_chapters(
    all_chapters: List[Dict[str, str]]
) -> None:
    if 'selected_chapters' not in st.session_state:
        st.session_state.selected_chapters = []
    
    chapter_display_names = [f"{chapter['chapter']} - {chapter['book']}" for chapter in all_chapters]
    
    def update_selected_chapters():
        st.session_state.selected_chapters = [
            chapter for chapter in all_chapters 
            if f"{chapter['chapter']} - {chapter['book']}" in st.session_state.chapter_selector
        ]

    selected_chapters = st.multiselect(
        'Select Chapter:', 
        options=chapter_display_names,
        default=[f"{chapter['chapter']} - {chapter['book']}" for chapter in st.session_state.selected_chapters if f"{chapter['chapter']} - {chapter['book']}" in chapter_display_names],
        key='chapter_selector',
        on_change=update_selected_chapters
    )
    
    new_selected_chapters = [
        chapter for chapter in all_chapters 
        if f"{chapter['chapter']} - {chapter['book']}" in selected_chapters
    ]
    
    if new_selected_chapters != st.session_state.selected_chapters:
        st.session_state.selected_chapters = new_selected_chapters

def _get_chapters_for_books(
    book_index: Dict[str, Any], 
    selected_books: List[str]
) -> List[Dict[str, str]]:
    all_chapters = []
    for book in selected_books:
        all_chapters.extend([{"book": book, "chapter": chapter} for chapter in book_index['chapters'][book]])
    return all_chapters

def handle_user_input(
    prompt: str,
    chat_history: list[dict],
    filtered_chapters: List[Dict[str, Any]]
) -> None:   
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching for sources..."):
        if not filtered_chapters and st.session_state.selected_books:
            filtered_chapters = _get_chapters_for_books(
                st.session_state.book_index,
                st.session_state.selected_books
            )

        retrieved_chapters = _retrieve_and_filter_chapters(filtered_chapters)

        similar_chunks = search_vector_db(
            question=prompt,
            dict_list=retrieved_chapters,
            history_messages=chat_history,
        )
        # Update sidebar content
        update_sidebar_content(similar_chunks)

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

def _retrieve_and_filter_chapters(
    filtered_chapters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
        
    filtered_key = "filtered_chapters_" + "_".join(sorted([f"{c['book']}_{c['chapter']}" for c in filtered_chapters]))

    if filtered_key not in st.session_state:
        st.session_state[filtered_key] = filter_by_criteria(
            st.session_state.index_list, 
            filtered_chapters, 
            {"book":"title", "chapter": "chapter"}
        )
    
    retrieved_chapters = st.session_state[filtered_key]

    return retrieved_chapters

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

def create_and_display_sidebar():
    with st.sidebar:
        st.title("Sources")
        
        if 'sidebar_content' not in st.session_state:
            st.session_state.sidebar_content = []
        
        # Create a container for the dynamic content
        sidebar_container = st.container()
        
        # Use st.empty() to create a placeholder that we can update
        sidebar_placeholder = st.empty()
    
    # Store the container and placeholder in session state
    st.session_state.sidebar_container = sidebar_container
    st.session_state.sidebar_placeholder = sidebar_placeholder
    
    logging.info("Sidebar created and displayed")

def update_sidebar_content(new_content):
    st.session_state.sidebar_content = new_content
    print(f"length of new content in update_sidebar_content: {len(new_content)}")
    
    # Update the sidebar content using the placeholder
    with st.session_state.sidebar_placeholder.container():
        if new_content:
            for i, source in enumerate(new_content):
                display_source_info(source, index=i)
        else:
            st.write("No relevant sources available.")

def display_source_info(
    source: dict,    
    index: int = None
) -> None:
    """
    Displays the source information in the Streamlit sidebar.

    Parameters:
    - source (dict): A dictionary containing source information.
    """

    top_text = f"""
    **Author:** {source['author']}

    **Chapter:** {source['chapter']}

    """
    is_paragraph = source['type'] == "paragraph"

    if is_paragraph:
        top_text += f"**Text:**\n"
    else:
        top_text += f"**AI Generated Summary:**\n"

    with st.sidebar:
        st.header(f"{index + 1}: {source['title']}")
        st.markdown(top_text)
        with st.expander(label="Expand Text", expanded=is_paragraph):
            st.markdown(source['text'])
        
# Main application setup
def main():
    PARAGRAPHS_FILEPATH = "./extracted_documents/all_books_paragraphs.json"
    BOOK_INDEX_FILEPATH = "./extracted_documents/book_and_chapter_index.json"
    
    st.set_page_config(
        page_title="Book Chat",
        page_icon=":book:",
        initial_sidebar_state="expanded"
    )
    
    
    if 'index_list' not in st.session_state:
        with st.spinner("Loading database..."):
            start_time = time.time()
            st.session_state.index_list = retrieve_file(PARAGRAPHS_FILEPATH)
            end_time = time.time()
            logging.info(f"\n\n{'-'*100}\nDatabase loaded successfully in {end_time - start_time} seconds\n\n")
    
    create_and_display_sidebar()
    select_books_and_chapters(BOOK_INDEX_FILEPATH)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "How may I help?"}]

    display_chat_history(st.session_state.chat_history)

    prompt = st.chat_input("What would you like to know?")

    if prompt:
        handle_user_input(
            prompt=prompt, 
            chat_history=st.session_state.chat_history, 
            filtered_chapters=st.session_state.selected_chapters,
        )

if __name__ == "__main__":
    main()