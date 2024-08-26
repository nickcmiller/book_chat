from dotenv import load_dotenv
load_dotenv()

from genai_toolbox.text_prompting.model_calls import openai_text_response, groq_text_response, fallback_text_response
from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding, find_similar_chunks
from genai_toolbox.helper_functions.string_helpers import retrieve_file
from genai_toolbox.chunk_and_embed.llms_with_queries import llm_response_with_query, stream_response_with_query

from typing import List, Optional, Dict, Any, Callable, Generator
import logging
import time
import json

def filter_by_criteria(
    dict_list: List[Dict[str, Any]],
    filter_list: List[Dict[str, Any]],
    field_mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
        Filters a list of dictionaries based on specified criteria.

        This function takes a list of dictionaries and filters it according to a list of filter criteria.
        Each filter criterion is a dictionary that specifies the conditions that must be met for an item
        in the input list to be included in the output list. The function uses a field mapping to determine
        which fields in the input dictionaries correspond to the fields in the filter criteria.

        Parameters:
        - dict_list (List[Dict[str, Any]]): The list of dictionaries to be filtered.
        - filter_list (List[Dict[str, Any]]): A list of dictionaries containing the filter criteria.
        - field_mapping (Dict[str, str]): A mapping of filter fields to the corresponding fields in the input dictionaries.

        Returns:
        - List[Dict[str, Any]]: A list of dictionaries that meet the specified filter criteria.

        Functionality:
            1. If the filter_list is empty, the function returns the original dict_list.
            2. It measures the time taken to perform the filtering operation for logging purposes.
            3. It iterates through each item in dict_list and checks if it matches any of the filter criteria.
            4. If an item matches, it is added to the result list.
            5. Finally, it logs the duration of the filtering operation and returns the filtered list.

        Example:
            dict_list = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
            filter_list = [{'name': 'Alice'}, {'age': 25}]
            field_mapping = {'name': 'name', 'age': 'age'}
            filtered = filter_by_criteria(dict_list, filter_list, field_mapping)
            # filtered will contain [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    """
    if not filter_list:
        return dict_list

    start_time = time.time()
    result = []
    for item in dict_list:
        if any(
            all(
                item.get(field_mapping[filter_field]) == filter_item.get(filter_field)
                for filter_field in filter_item
                if filter_field in field_mapping and filter_item.get(filter_field) is not None
            )
            for filter_item in filter_list
        ):
            result.append(item)

    end_time = time.time()
    logging.info(f"Filtering by criteria duration: {end_time - start_time} seconds\n")

    return result

def format_messages(
    history_messages: List[Dict[str, Any]],
    system_instructions: str = None, 
    history_limit: Optional[int] = 6
) -> List[Dict[str, str]]:
    """
        Formats the chat history messages for use in a conversational AI model.

        This function takes a list of history messages and formats them according to the specified system instructions
        and history limit. It ensures that the most relevant messages are included in the final output, which is 
        essential for maintaining context in conversations.

        Parameters:
        - history_messages (List[Dict[str, Any]]): A list of dictionaries representing the chat history, 
          where each dictionary contains the role (user or assistant) and the content of the message.
        - system_instructions (str, optional): Instructions for the model to follow while generating the response.
        - history_limit (Optional[int]): The maximum number of history messages to include in the formatted messages. 
          If None, all messages are included. Defaults to 6.

        Returns:
        - List[Dict[str, str]]: A list of formatted messages ready for use in the AI model, including the system 
          instructions if provided.

        Example:
            history_messages = [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"},
                {"role": "user", "content": "I need assistance with my order."}
            ]
            formatted = format_messages(history_messages, system_instructions="You are a helpful assistant.")
            # formatted will contain the system instructions followed by the last 6 messages from history.
    """

    if history_limit is None:
        history_limit = len(history_messages)
    else:
        history_limit = int(history_limit)
    messages_to_use = history_messages[-history_limit:]

    formatted_messages = []

    if system_instructions:
        formatted_messages.append({
            "role": "system", 
            "content": system_instructions
        })
        for message in messages_to_use:
            if message["role"] != "system":
                formatted_messages.append(message)
    else:
        formatted_messages.extend(messages_to_use)

    return formatted_messages

def fallback_query(
    prompt: str,
    system_instructions: str = None,
    history_messages: List[Dict[str, Any]] = None,
    history_limit: Optional[int] = 6
) -> str:
    """
        Fallback Query Function

        This function is responsible for generating a response based on a given prompt, system instructions, 
        and chat history messages. It formats the messages appropriately for the OpenAI API and attempts to 
        retrieve a response from a series of fallback models in a specified order. If an error occurs during 
        the response retrieval, it logs the error and returns a default message indicating that the query 
        could not be understood.

        Parameters:
        - prompt (str): The prompt that guides the model's response generation.
        - system_instructions (str, optional): Instructions for the model to follow while generating the response.
        - history_messages (List[Dict[str, Any]], optional): A list of previous messages in the chat history 
          to provide context for the response.
        - history_limit (Optional[int], optional): The maximum number of history messages to include in the 
          formatted messages. If None, all messages are included. Defaults to 6.

        Returns:
        - str: The generated response from the model, or a default error message if an exception occurs.

        Example:
            response = fallback_query(
                prompt="What is the capital of France?",
                system_instructions="Provide a concise answer.",
                history_messages=[{"role": "user", "content": "Tell me about France."}],
                history_limit=3
            )
            # response will contain the model's answer to the prompt.
    """
    if history_messages is None:
        history_messages = []

    formatted_messages = format_messages(
        system_instructions=system_instructions, 
        history_messages=history_messages,
        history_limit=history_limit
    )

    fallback_model_order = [
        {
            "provider": "groq", 
            "model": "llama3.1-70b"
        },
        {
            "provider": "openai", 
            "model": "4o-mini"
        },
        {
            "provider": "anthropic", 
            "model": "sonnet"
        }
    ] 

    try:
        response = fallback_text_response(
            prompt, 
            system_instructions, 
            formatted_messages,
            model_order=fallback_model_order
        )
        return response
    except Exception as e:
        logging.error(f"Error: {e}")
        return "Sorry, I don't understand."

def groq_query(
    prompt: str, 
    system_instructions: str = None, 
    history_messages: List[Dict[str, Any]] = None,
    model_choice: str = "llama3.1-70b"
) -> str:
    """
        Executes a query using the Groq model to generate a response based on the provided prompt and chat history.

        Parameters:
        - prompt (str): The user prompt that guides the model's response.
        - system_instructions (str, optional): Instructions for the model to follow while generating the response.
        - history_messages (List[Dict[str, Any]], optional): A list of previous messages in the chat history to provide context.
        - model_choice (str, optional): The specific model to use for generating the response. Defaults to "llama3.1-70b".

        Returns:
        - str: The generated response from the Groq model.

        This function formats the input messages, calls the Groq model, and handles any exceptions that may occur during the process.
    """
    if history_messages is None:
        history_messages = []

    formatted_messages = format_messages(
        system_instructions=system_instructions, 
        history_messages=history_messages,
        history_limit=4
    )

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

def search_vector_db(
    question: str,
    dict_list: List[Dict[str, Any]],
    history_messages: List[Dict[str, str]],
    similarity_threshold: float = 0.3,
    filter_limit: int = 15,
    max_similarity_delta: float = 0.075,
) -> List[Dict[str, Any]]:
    """
        Searches the vector database for similar chunks of text based on the provided question and chat history.

        This function first creates a query for the vector database using the provided question and chat history. 
        It then retrieves similar chunks of text from the database based on the generated query, logging the duration 
        of each operation for performance monitoring. The function is designed to help in finding relevant information 
        that can assist in answering the user's question effectively.

        Parameters:
        - question (str): The user's question that needs to be answered.
        - dict_list (List[Dict[str, Any]]): A list of dictionaries representing the data in the vector database.
        - history_messages (List[Dict[str, str]]): A list of dictionaries representing the chat history, 
          where each dictionary contains the role (user or assistant) and the content of the message.
        - similarity_threshold (float, optional): The minimum similarity score for a chunk to be considered relevant. 
          Defaults to 0.3.
        - filter_limit (int, optional): The maximum number of similar chunks to return. Defaults to 15.
        - max_similarity_delta (float, optional): The maximum allowable difference in similarity scores for filtering. 
          Defaults to 0.075.

        Returns:
        - List[Dict[str, Any]]: A list of dictionaries representing the similar chunks retrieved from the vector database.
    """
    start_time = time.time()
    vectordb_query = _create_vectordb_query(question, history_messages)
    vectordb_end_time = time.time()
    
    logging.info(f"Vectordb duration: {vectordb_end_time - start_time} seconds\n")

    similar_chunks = _retrieve_similar_chunks(
        dict_list, 
        vectordb_query, 
        similarity_threshold=similarity_threshold,
        filter_limit=filter_limit,
        max_similarity_delta=max_similarity_delta,
    )
    
    similar_chunks_end_time = time.time()
    logging.info(f"Similar chunk retrieval duration: {similar_chunks_end_time - vectordb_end_time} seconds\n")

    return similar_chunks

def _create_vectordb_query(
    question: str,
    history_messages: List[Dict[str, str]], 
) -> str:

    context_prompt = f"""
        Request: {question}\n\nBased on this request, provide context for the query that I should make to my vector database.
        Use chat history to thoroughly establish the intent of the question, the people referenced by the question, and terms, topics, individuals, or themes targeted by the question in chat history.
        If any relevant topics, themes, terms, or individuals mentioned in chat history, incorporate them in the request in detail.
        Significantly lengthen the request and include as many contextual details as possible to enhance the relevance of the query.
        Only return the request. Don't preface it or provide an introductory message.
    """

    context_system_instructions = "You are an expert on adding context to requests asked to a vector database containing chunks of transcripts. You add sub-questions and contextual details to make the query more specific and relevant to the chat history."

    context = fallback_query(
        prompt=context_prompt, 
        system_instructions=context_system_instructions, 
        history_messages=history_messages,
        history_limit=8
    )

    vectordb_query = f"{question}\n{context}"

    logging.info(f"Vectordb query: {vectordb_query}")

    return vectordb_query

def _retrieve_similar_chunks(
    extracted_list: List[Dict[str, Any]],
    query: str, 
    similarity_threshold: float,
    filter_limit: int,
    max_similarity_delta: float,
    embedding_function: Callable = create_openai_embedding, 
    model_choice: str = "text-embedding-3-large"
) -> List[Dict[str, Any]]:
    """
        This function retrieves and identifies chunks of text that are similar to a given query based on their embeddings.

        Parameters:
        - file_path (str): The path to the file containing the data from which chunks will be extracted.
        - query (str): The input query string for which similar chunks are to be found.
        - similarity_threshold (float): The minimum similarity score for a chunk to be considered similar.
        - filter_limit (int): The maximum number of similar chunks to return.
        - max_similarity_delta (float): The maximum allowed difference in similarity between the most similar chunk and the others to be included in the results.
        - embedding_function (Callable): A callable function that generates embeddings for the given text. Defaults to create_openai_embedding.
        - model_choice (str): The model to be used for generating embeddings. Defaults to "text-embedding-3-large".

        Returns:
        - List[Dict[str, Any]]: A list of dictionaries containing the similar chunks, each with an added 'similarity' key indicating the similarity score.

        Functionality:
            1. Extracts data from the specified file using the retrieve_file function.
            2. Computes embeddings for the query and compares them against the extracted chunks using the find_similar_chunks function.
            3. Logs the number of similar chunks retrieved and any warnings if no similar chunks are found.
            4. Returns the list of similar chunks that meet the similarity criteria.

        This function is essential for enabling the system to respond to user queries by identifying relevant information based on semantic similarity.
    """
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
        logging.warning(f"No similar rows found for the query embedding over similarity threshold of {similarity_threshold}.")
        return []

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

        1. History Message Preparation: It takes the last few messages from the chat history and 
        prepends a system message to provide context for the query.

        2. Vector Database Query Creation: It constructs a query specifically tailored for the vector 
        database using the provided question and the prepared history messages. This helps in understanding 
        the context and intent behind the user's question.

        3. Retrieving Similar Chunks: The function then calls another function to retrieve similar chunks 
        of text from the specified file based on the generated vector database query. This step is crucial 
        for finding relevant information that can help answer the user's question.

        4. Query Revision: After retrieving similar chunks, it revises the original question to enhance 
        its clarity and relevance based on the chat history. This ensures that the query is as effective as 
        possible in eliciting a useful response.

        5. Generating the Answer: Finally, it generates an answer using the revised query, the history 
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
    
    new_query = _revise_query(question, history_messages)

    new_query_end_time = time.time()
    logging.info(f"New query duration: {new_query_end_time - start_time} seconds\n\n\n")

    return _generate_answer(
        new_query, 
        history_messages, 
        similar_chunks
    )

def _revise_query(
    question: str,
    history_messages: List[Dict[str, str]], 
) -> str:

    context_prompt = f"""
        Succinctly and briefly describe how the question is related to the chat history.
        ---
        Question: {question}
    """
    context_system_instructions = "You are an expert research assistant who helps contextualize questions using chat history. The rewrite should be concise and direct. It should end with a ? mark. If the user requests a format or style, include that requested format or style in the rewrite. Example: If the user asks for an outline, include the request for an outline in the rewrite. Only return the request. Don't preface it or provide an introductory message."

    context = fallback_query(
        prompt=context_prompt,
        system_instructions=context_system_instructions,
        history_messages=history_messages,
        history_limit=6
    )

    new_query = f"""
        Question: {question}
        ---
        Context for Question: {context}
        ---
    """

    logging.info(f"Revised query: {new_query}")

    return new_query

def _generate_answer(
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

    formatted_history_messages = format_messages(
        history_messages=history_messages,
        system_instructions=llm_system_prompt,
        history_limit=6
    )

    source_template = """
    Book: *{title}*
    Chapter: {chapter}
    Author: {author}

    Text: {text}
    """

    return stream_response_with_query(
        similar_chunks,
        question=query,
        history_messages=formatted_history_messages,
        source_template=source_template,
        template_args={
            "title": "title", 
            "text": "text", 
            "author": "author", 
            "chapter": "chapter", 
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

if __name__ == "__main__":
    file_path = "./extracted_documents/all_books_paragraphs.json"
    extracted_list = retrieve_file(file_path)
    if not extracted_list:
        logging.error(f"No data extracted from file: {file_path}")

    print(f"Extracted list length: {len(extracted_list)}")

    unique_values = get_unique_values(extracted_list, "chapter")
    print(f"Unique values: {json.dumps(unique_values, indent=2)}")



    # history_messages = [
    #     {"role": "system", "content": "Why are babies so smart?"},
    #     {"role": "assistant", "content": "Babies are born with a certain level of intelligence, but it depends on the environment and the individual's upbringing."},
    # ]
    # query = "Can you go into more detail?"

    # response = query_data(query, file_path, history_messages)
    # for chunk in response:
    #     print(chunk)