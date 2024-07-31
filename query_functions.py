from dotenv import load_dotenv
load_dotenv()

from genai_toolbox.text_prompting.model_calls import openai_text_response
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding, find_similar_chunks
from genai_toolbox.helper_functions.string_helpers import retrieve_file
from genai_toolbox.chunk_and_embed.llms_with_queries import llm_response_with_query
from typing import List, Optional, Dict, Any, Callable
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

def retrieve_similar_chunks(
    file_path: str,
    query: str, 
    embedding_function: Callable = create_openai_embedding, 
    model_choice: str = "text-embedding-3-large", 
    threshold: float = 0.4, 
    max_returned_chunks: int = 15
) -> List[Dict[str, Any]]:
    extracted_list = retrieve_file(file_path)
    similar_chunks = find_similar_chunks(
        query, 
        extracted_list, 
        embedding_function=embedding_function, 
        model_choice=model_choice, 
        threshold=threshold, 
        max_returned_chunks=max_returned_chunks
    )
    return similar_chunks

def process_query(
    file_path: str, 
    query: str, 
) -> Dict[str, Any]:  # Change return type to Dict[str, Any]
    similar_chunks = retrieve_similar_chunks(file_path, query)
    print(f"Number of similar chunks: {len(similar_chunks)}")
    
    if not similar_chunks:
        return {"llm_response": "No relevant information found for the given query."}

    for chunk in similar_chunks:
        print(f"\n\nText: {chunk['text']}\n\nSimilarity score: {chunk['similarity']}")
        print(chunk.keys())

    llm_system_prompt_default = f"""
    Use numbered references (e.g. [1]) to cite the sources that are given to you in your answers.
    List the references used at the bottom of your answer.
    Do not refer to the source material in your text, only in your number citations
    Give a detailed answer.
    
    Example:
    Text that is referring to the first source.[1] Text that cites sources 2 and 3.[2][3]

    **References:**
        1. Last Name, First Name. "Chapter Title." *Book Title*, Publisher, Year.
        2. Rowling, J.K. "The Forbidden Forest." *The Philosophers's Stone.*, Bloomsbury, 2005.
        3. Rowling, J.K. "Dobby's Warning." *The Chamber of Secrets.*, Bloomsbury, 2005.
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
    print(f"\n\nprocess_query response: {response['llm_response']}\n\n")
    return response  # Return the entire response dictionary

def query_data(
    file_path: str, 
    history_messages: List[Dict[str, str]], 
    query: str
) -> str:
    system_instructions = """
    Based on the chat history available to you, please re-write this question to include relevant context.
    It will be embedded and used in a similarity search to find relevant information.
    Do not respond to the user, rephrase the question for a search. 
    The response should be a question. The response should be a single sentence.
    """

    new_query = openai_query(query, system_instructions=system_instructions, history_messages=history_messages)
    print(f"\n\nnew_query: {new_query}\n\n")
    response = process_query(file_path, new_query)
    print(f"\n\nquery_data response: {response['llm_response']}\n\n")
    return response['llm_response']

if __name__ == "__main__":
    file_path = "./extracted_documents/The_Philosophical_Baby_all_paragraphs.json"
    history_messages = [
        {"role": "system", "content": "Why are babies so smart?"},
        {"role": "assistant", "content": "Babies are born with a certain level of intelligence, but it depends on the environment and the individual's upbringing."},
    ]
    query = "Can you go into more detail?"

    query_data(file_path, history_messages, query)

