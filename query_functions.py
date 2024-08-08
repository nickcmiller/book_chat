from dotenv import load_dotenv
load_dotenv()

from genai_toolbox.text_prompting.model_calls import openai_text_response, groq_text_response
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding, find_similar_chunks
from genai_toolbox.helper_functions.string_helpers import retrieve_file
from genai_toolbox.chunk_and_embed.llms_with_queries import llm_response_with_query, stream_response_with_query
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

def groq_query(
    prompt: str, 
    system_instructions: str = None, 
    history_messages: List[Dict[str, Any]] = None
) -> str:
    """
    Query the Groq API for a response based on the provided prompt and history.
    """
    if history_messages is None:
        history_messages = []

    formatted_messages = format_messages(prompt, system_instructions, history_messages)
    logging.info(f"Formatted messages for Groq query: {formatted_messages}")

    try:
        response = groq_text_response(
            prompt, 
            system_instructions, 
            formatted_messages,
            model_choice="llama3.1-70b"
        )
        return response
    except Exception as e:
        logging.error(f"Error: {e}")
        return "Sorry, I don't understand."

def retrieve_similar_chunks(
    file_path: str,
    query: str, 
    embedding_function: Callable = create_openai_embedding, 
    model_choice: str = "text-embedding-3-large", 
    threshold: float = 0.4, 
    max_returned_chunks: int = 15
) -> List[Dict[str, Any]]:
    extracted_list = retrieve_file(file_path)
    if not extracted_list:
        logging.error(f"No data extracted from file: {file_path}")
        return []

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
) -> Dict[str, Any]:
    """
    Processes a query by retrieving similar chunks from a specified file and generating a response.

    This function takes a file path and a query string as input, retrieves similar chunks of text 
    from the file using the `retrieve_similar_chunks` function, and then formats a response based 
    on the retrieved chunks. If no similar chunks are found, it returns a message indicating that 
    no relevant information was found.

    Args:
        file_path (str): The path to the file from which to retrieve text chunks.
        query (str): The query string for which similar text chunks are to be found.

    Returns:
        Dict[str, Any]: A dictionary containing the LLM response. If similar chunks are found, 
        the response will be based on the content of those chunks. If no chunks are found, the 
        response will indicate that no relevant information was found.

    Example:
        result = process_query("./path/to/file.json", "What is the significance of the Forbidden Forest?")
        print(result)  # Output will depend on the contents of the file and the query.
    """
    similar_chunks = retrieve_similar_chunks(file_path, query)
    print(f"Number of similar chunks: {len(similar_chunks)}")
    
    if not similar_chunks:
        return {"llm_response": "No relevant information found for the given query."}

    for chunk in similar_chunks:
        print(f"\n\nText:\n{chunk['text']}\nSimilarity score: {chunk['similarity']}")
        print(chunk.keys())

    llm_system_prompt = f"""
    Use numbered references (e.g. [1]) to cite the chapters that are given to you in your answers.
    List the references used at the bottom of your answer. Only list each chapter once.
    Do not refer to the source material in your text, only in your number citations
    
    Example:
    ```
    Text that is referring to the first source.[1] Text that cites sources 2 and 3.[2][3]
    
    Text that cites source 1 for a second time.[1]

    **References:**
        1. "Chapter Title", *Book Title* by Author Name
        2. "The Forbidden Forest", *The Philosophers's Stone* by J.K. Rowling
        3. "Dobby's Warning", *The Chamber of Secrets* by J.K. Rowling
    ```
    Make sure the chapters are included in the references.
    If the same chapter is used multiple times, only include it once in the references.
    Give a detailed answer.
    """

    source_template = """
    Book: *{title}*
    Chapter: {chapter}
    Author: {author}
    Publisher: {publisher}

    Text: {text}
    """

    return stream_response_with_query(
        similar_chunks,
        question=query,
        llm_system_prompt=llm_system_prompt,
        source_template=source_template,
        template_args={
            "title": "title", 
            "text": "text", 
            "author": "author", 
            "chapter": "chapter", 
            "publisher": "publisher"
        },
        llm_model_order=[
            {
                "provider": "groq", 
                "model": "llama3.1-70b"
            },
            {
                "provider": "groq", 
                "model": "llama3.1-405b"
            },
        ],
    )

def query_data(
    question: str,
    file_path: str, 
    history_messages: List[Dict[str, str]], 
) -> str:
    revision_prompt = f"""
        Question: {question}
        When possible, rewrite the question using <chat history> to identify the intent of the question, the people referenced by the question, and ideas / topics / themes targeted by the question in <chat history>.
        If the <chat history> does not contain any information about the people, ideas, or topics relevant to the question, then do not make any assumptions.
        Only return the request. Don't preface it or provide an introductory message.
    """
    revision_system_instructions = "You are an assistant that concisely and carefully rewrites questions. The less than (<) and greater than (>) signs are telling you to refer to the chat history. Don't use < or > in your response."

    new_query = groq_query(
        prompt=revision_prompt, 
        system_instructions=revision_system_instructions, 
        history_messages=history_messages
    )
    print(f"\n\nnew_query:\n{new_query}\n\n")
    return process_query(file_path, new_query)

if __name__ == "__main__":
    file_path = "./extracted_documents/all_books_paragraphs.json"
    history_messages = [
        {"role": "system", "content": "Why are babies so smart?"},
        {"role": "assistant", "content": "Babies are born with a certain level of intelligence, but it depends on the environment and the individual's upbringing."},
    ]
    query = "Can you go into more detail?"

    response = query_data(query, file_path, history_messages)
    for chunk in response:
        print(chunk)