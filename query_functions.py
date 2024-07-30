from dotenv import load_dotenv
load_dotenv()

from genai_toolbox.text_prompting.model_calls import openai_text_response
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding, find_similar_chunks
from genai_toolbox.helper_functions.string_helpers import retrieve_file
from genai_toolbox.chunk_and_embed.llms_with_queries import llm_response_with_query
from typing import List, Optional, Dict, Any
import logging

def format_messages(
    prompt: str, 
    system_instructions: str, 
    history_messages: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Format the messages for the OpenAI API."""
    formatted_messages = []
    if system_instructions:
        formatted_messages.append({"role": "system", "content": system_instructions})
    formatted_messages.extend(history_messages)
    formatted_messages.append({"role": "user", "content": prompt})
    return formatted_messages

def handle_openai_response(
    prompt: str, 
    system_instructions: str, 
    formatted_messages: List[Dict[str, str]]
) -> str:
    """
    Handle the OpenAI API response.
    """
    try:
        response = openai_text_response(prompt, system_instructions, formatted_messages)
        logging.info(f"Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error: {e}")
        return "Sorry, I don't understand."

def openai_query(
    prompt: str, 
    system_instructions: str = None, 
    history_messages: List[Dict[str, Any]] = None
) -> str:
    """
    Query the OpenAI API for a response based on the provided prompt and history.
    """
    if history_messages is None:
        history_messages = []

    formatted_messages = format_messages(prompt, system_instructions, history_messages)
    logging.info(f"Formatted messages for OpenAI query: {formatted_messages}")
    return handle_openai_response(prompt, system_instructions, formatted_messages)


if __name__ == "__main__":
    # print("Testing OpenAI query...")
    # system_instructions = "Respond as if you were Ozzy Osbourne."
    # history_messages = [{"role": "user", "content": "Hello, how are you?"}, {"role": "assistant", "content": "I'm not feeling too good today."}]
    # print(openai_query(
    #     "Why is that?", 
    #     system_instructions=system_instructions, 
    #     history_messages=history_messages
    # ))
    file_path = "./extracted_documents/The_Philosophical_Baby_all_paragraphs.json"
    extracted_list = retrieve_file(file_path)
    query="When makes babies intelligent?"
    similar_chunks = find_similar_chunks(
        query, 
        extracted_list, 
        embedding_function=create_openai_embedding, 
        model_choice="text-embedding-3-large", 
        threshold=0.4, 
        max_returned_chunks=15
    )
    print(f"Number of similar chunks: {len(similar_chunks)}")
    for chunk in similar_chunks:
        print(f"\n\nText: {chunk['text']}\n\nSimilarity score: {chunk['similarity']}")
        print(chunk.keys())

    llm_system_prompt_default = f"""
    Use numbered references (e.g. [1]) to cite the sources that are given to you in your answers.
    List the references used at the bottom of your answer.
    Use MLA Citation Style that references the chapter.
    Do not refer to the source material in your text, only in your number citations
    Give a detailed answer.
    """


    source_template = """
    Book: *{title}*
    Chapter: {chapter}
    Author: {author}
    Publisher: {publisher}

    Text: {text}
    """

    response = llm_response_with_query(
        similar_chunks,
        question=query,
        llm_system_prompt=llm_system_prompt_default,
        source_template=source_template,
        template_args={"title": "title", "text": "text", "author": "author", "chapter": "chapter", "publisher": "publisher"}
    )
    print(response['llm_response'])

