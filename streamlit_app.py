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

        import json
        print("Selected Chapters:", json.dumps(st.session_state.selected_chapters, indent=2))

    selected_chapters = st.multiselect(
        'Select Chapter:', 
        options=chapter_display_names,
        default=[f"{chapter['chapter']} - {chapter['book']}" for chapter in st.session_state.selected_chapters if f"{chapter['chapter']} - {chapter['book']}" in chapter_display_names],
        key='chapter_selector',
        on_change=update_selected_chapters
    )
    
    # Update session state only if the selection has changed
    new_selected_chapters = [
        chapter for chapter in all_chapters 
        if f"{chapter['chapter']} - {chapter['book']}" in selected_chapters
    ]
    
    if new_selected_chapters != st.session_state.selected_chapters:
        st.session_state.selected_chapters = new_selected_chapters

        import json
        print("Selected Books:", json.dumps(st.session_state.selected_books, indent=2))
        # st.rerun()

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
    paragraphs_filepath: str,
    filtered_chapters: List[Dict[str, Any]]
) -> None:
    
    st.sidebar.empty() 
    st.sidebar.title("Sources")
   
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching for sources..."):
        # If no chapters are selected, use all chapters from selected books
        if not filtered_chapters and st.session_state.selected_books:
            print(f"Selected books: {st.session_state.selected_books}")
            filtered_chapters = _get_chapters_for_books(
                st.session_state.book_index,
                st.session_state.selected_books
            )
            print(f"filtered_chapters after none selected: {len(filtered_chapters)}")

        retrieved_chapters = _retrieve_and_filter_chapters(paragraphs_filepath, filtered_chapters)

        similar_chunks = search_vector_db(
            question=prompt,
            dict_list=retrieved_chapters,
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

def _retrieve_and_filter_chapters(
    paragraphs_filepath: str,
    filtered_chapters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    print(f"filtered_chapters input: {len(filtered_chapters)}")
    print(f"filtered_chapters keys: {filtered_chapters[0].keys()}")
    print(f"filtered_chapter: {json.dumps(filtered_chapters, indent=2)}")
    
    if 'index_list' not in st.session_state:
        st.session_state.index_list = retrieve_file(paragraphs_filepath)
    
    print(f"index_list: {len(st.session_state.index_list)}")
    
    filtered_key = "filtered_chapters_" + "_".join(sorted([f"{c['book']}_{c['chapter']}" for c in filtered_chapters]))
    print(f"filtered_key: {filtered_key}")

    if filtered_key not in st.session_state:
        st.session_state[filtered_key] = filter_by_criteria(
            st.session_state.index_list, 
            filtered_chapters, 
            {"book":"title", "chapter": "chapter"}
        )
    
    retrieved_chapters = st.session_state[filtered_key]
    print(f"retrieved_chapters: {len(retrieved_chapters)}")
    
    # Add this section to verify the filtered results
    for chapter in retrieved_chapters[:5]:  # Print first 5 chapters for verification
        print(f"Book: {chapter['title']}, Chapter: {chapter['chapter']}")

    return retrieved_chapters

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

# Main application setup
def main():
    st.set_page_config(
        page_title="Book Chat",
        page_icon=":book:",
        initial_sidebar_state="expanded"
    )

    path_to_book_index = "./extracted_documents/book_and_chapter_index.json"
    select_books_and_chapters(path_to_book_index)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "How may I help?"}]

    display_chat_history(st.session_state.chat_history)

    prompt = st.chat_input("What would you like to know?")

    if prompt:
        handle_user_input(
            prompt=prompt, 
            chat_history=st.session_state.chat_history, 
            paragraphs_filepath="./extracted_documents/all_books_paragraphs.json",
            filtered_chapters=st.session_state.selected_chapters,
        )

if __name__ == "__main__":
    main()