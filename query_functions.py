from dotenv import load_dotenv
load_dotenv()

from genai_toolbox.text_prompting.model_calls import openai_text_response, groq_text_response
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding, find_similar_chunks
from genai_toolbox.helper_functions.string_helpers import retrieve_file
from genai_toolbox.chunk_and_embed.llms_with_queries import llm_response_with_query, stream_response_with_query

from typing import List, Optional, Dict, Any, Callable, Generator
import logging
import time

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
    history_messages: List[Dict[str, Any]] = None,
    model_choice: str = "llama3.1-70b"
) -> str:
    """
    Query the Groq API for a response based on the provided prompt and history.
    """
    if history_messages is None:
        history_messages = []

    formatted_messages = format_messages(prompt, system_instructions, history_messages)
    try:
        response = groq_text_response(
            prompt, 
            system_instructions, 
            formatted_messages,
            model_choice=model_choice
        )
        return response
    except Exception as e:
        logging.error(f"Error: {e}")
        return "Sorry, I don't understand."

def create_vectordb_query(
    question: str,
    history_messages: List[Dict[str, str]], 
) -> str:

    vectordb_prompt = f"""
        Request: {question}\n\nBased on this request, what request should I make to my vector database?
        Use prior messages to establish the intent and context of the question. 
        If any relevant topics, themes, or individuals mentioned in chat history, incorporate them in the request.
        Significantly lengthen the request and include as many contextual details as possible to enhance the relevance of the query.
        Only return the request. Don't preface it or provide an introductory message.
    """

    vectordb_system_instructions = "You expand on questions asked to a vector database containing chunks of transcripts. You add sub-questions and contextual details to make the query more specific and relevant to the chat history."

    vectordb_query = groq_query(
        prompt=vectordb_prompt, 
        system_instructions=vectordb_system_instructions, 
        history_messages=history_messages
    )

    logging.info(f"Vectordb query: {vectordb_query}")

    return vectordb_query

def retrieve_similar_chunks(
    file_path: str,
    query: str, 
    similarity_threshold: float,
    filter_limit: int,
    max_similarity_delta: float,
    embedding_function: Callable = create_openai_embedding, 
    model_choice: str = "text-embedding-3-large"
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
        similarity_threshold=similarity_threshold,
        filter_limit=filter_limit,
        max_similarity_delta=max_similarity_delta
    )

    logging.info(f"Retrieved {len(similar_chunks)} similar chunks")

    if not similar_chunks:
        logging.warning(f"No similar rows found for the query embedding.")
        return []

    return similar_chunks

def revise_query(
    question: str,
    history_messages: List[Dict[str, str]], 
) -> str:
    revision_prompt = f"""
        When possible, rewrite the question using <chat history> to identify the intent of the question, the people referenced by the question, and ideas / topics / themes targeted by the question in <chat history>.
        If the <chat history> does not contain any information about the people, ideas, or topics relevant to the question, then do not make any assumptions.
        The rewrite should be concise and direct. It should end with a ? mark.
        If the user requests a format or style, include that requested format or style in the rewrite.
        Only return the request. Don't preface it or provide an introductory message.

        ---
        Question: {question}
        ---
    """
    revision_system_instructions = "You are an assistant that concisely and carefully rewrites questions. The less than (<) and greater than (>) signs are telling you to refer to the chat history. Don't use < or > in your response."

    new_query = groq_query(
        prompt=revision_prompt,
        system_instructions=revision_system_instructions, 
        history_messages=history_messages
    )

    logging.info(f"Revised query: {new_query}")

    return new_query

def generate_answer(
    query: str,
    history_messages: List[Dict[str, str]],
    similar_chunks: List[Dict[str, Any]],
) -> Generator[str, Any, Any]:
    """
        Generates an answer based on the provided query, chat history, and similar chunks of text.

        Parameters:
        - query (str): The user's question or prompt that needs to be answered.
        - history_messages (List[Dict[str, str]]): A list of messages representing the chat history, 
        where each message is a dictionary containing the role (user or assistant) and the content.
        - similar_chunks (List[Dict[str, Any]]): A list of chunks of text that are similar to the query, 
        which will be used to formulate the answer.

        Returns:
        - Generator[str, Any, Any]: A generator that yields the generated answer as a string. 
        The answer will include citations to the sources used, formatted according to the specified 
        citation style in the system prompt.

        The function first constructs a system prompt that instructs the language model to use numbered 
        references for citing chapters from the provided sources. It then prepares the chat history 
        for the model by taking the last few messages and inserting the system prompt at the beginning.

        A source template is defined to format the information about each source, including the title, 
        chapter, author, and publisher. Finally, the function calls `stream_response_with_query` to 
        generate the answer, passing in the necessary parameters, including the query, chat history, 
        source template, and model configuration.

        If no similar chunks are found, the function will inform the user that no sources were found.
    """
    llm_system_prompt = f"""
        Use numbered references (e.g. [1]) to cite the chapters that are given to you.
        If the same source is used multiple times, refer to the same number for citations.
        Do not refer to the source material in your text, only in your number citations.

        Example:
        ```
        Text that is referring to the first source.[1] Text that cites sources 2 and 3.[2][3]
        
        Text that cites source 1 for a second time.[1]
        ```
        Give a thorough, detailed, in-depth answer.

        If there are no sources, then tell me 'No sources found'.
    """

    revised_history_messages = history_messages[-4:]
    revised_history_messages.insert(0, {"role": "system", "content": llm_system_prompt})

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
        history_messages=revised_history_messages,
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
                "provider": "openai", 
                "model": "4o-mini"
            },
            {
                "provider": "openai", 
                "model": "4o"
            },
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

def search_vector_db(
    question: str,
    file_path: str,
    history_messages: List[Dict[str, str]], 
    similarity_threshold: float = 0.4,
    filter_limit: int = 15,
    max_similarity_delta: float = 0.075,
) -> str:
    start_time = time.time()
    
    vectordb_query = create_vectordb_query(question, history_messages)
    vectordb_end_time = time.time()
    logging.info(f"\nVectordb duration: {vectordb_end_time - start_time} seconds\n")

    similar_chunks = retrieve_similar_chunks(
        file_path, 
        vectordb_query, 
        similarity_threshold=similarity_threshold,
        filter_limit=filter_limit,
        max_similarity_delta=max_similarity_delta,
    )
    
    similar_chunks_end_time = time.time()
    logging.info(f"\nSimilar chunk retrieval duration: {similar_chunks_end_time - vectordb_end_time} seconds\n")

    return similar_chunks

def query_data(
    question: str,
    similar_chunks: List[Dict[str, Any]],
    history_messages: List[Dict[str, str]], 
) -> str:
    """
        Queries the data based on the provided question and history messages.

        This function serves as the main entry point for querying the vector database and retrieving relevant 
        information based on the user's question. It performs the following steps:

        1. **History Message Preparation**: It takes the last few messages from the chat history and 
        prepends a system message to provide context for the query.

        2. **Vector Database Query Creation**: It constructs a query specifically tailored for the vector 
        database using the provided question and the prepared history messages. This helps in understanding 
        the context and intent behind the user's question.

        3. **Retrieving Similar Chunks**: The function then calls another function to retrieve similar chunks 
        of text from the specified file based on the generated vector database query. This step is crucial 
        for finding relevant information that can help answer the user's question.

        4. **Query Revision**: After retrieving similar chunks, it revises the original question to enhance 
        its clarity and relevance based on the chat history. This ensures that the query is as effective as 
        possible in eliciting a useful response.

        5. **Generating the Answer**: Finally, it generates an answer using the revised query, the history 
        messages, and the similar chunks retrieved. The function returns both the generated answer and the 
        list of similar chunks for further processing or display.

        Parameters:
        - question (str): The question to be asked.
        - file_path (str): The path to the file containing the data.
        - history_messages (List[Dict[str, str]]): A list of dictionaries representing the chat history, 
        where each dictionary contains the role (user or assistant) and the content of the message.

        Returns:
        - Tuple[str, List[Dict[str, str]]]: A tuple containing the generated answer as a string and 
        a list of similar chunks retrieved from the data source.
    """

    start_time = time.time()
    
    new_query = revise_query(question, history_messages)

    new_query_end_time = time.time()
    logging.info(f"\nNew query duration: {new_query_end_time - start_time} seconds\n")

    return generate_answer(
        new_query, 
        history_messages, 
        similar_chunks
    )

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