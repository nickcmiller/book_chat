from typing import List, Dict, Any 
import tiktoken

from dotenv import load_dotenv
load_dotenv()

from genai_toolbox.text_prompting.model_calls import fallback_text_response

def summarize_with_revisions(
    text: str,
    prompt_list: List[Dict[str, Any]]
) -> str:
    
    current_context = ""
    for p in prompt_list:
        prompt = p['prompt']
        model_order = p['model_order']
        system_instructions = p['system_instructions']

        structured_prompt = _create_structured_prompt(prompt, current_context, text)

        current_summary = fallback_text_response(
            prompt=structured_prompt,
            system_instructions=system_instructions,
            model_order=model_order
        )
        current_context = f"""
            {current_summary}
        """
    return current_summary

def _create_structured_prompt(
    prompt: str, 
    context: str, 
    text: str
) -> str:
    prompt = f"""
        {prompt}
        {context}
        Chapter Text:
        ```
        {text}
        ```
    """
    return prompt

def summarize_chapter(
    chapter_text: str, 
    chapter_title: str, 
    metadata: Dict[str, Any]
):
    if len(chapter_text) < 100 or chapter_text is None:
        return None

    chapter=chapter_title, 
    author=metadata['creator'], 
    title=metadata['title'],
    publisher=metadata['publisher'],
    
    system_instructions = """
        Adhere to the following formatting rules when creating the outline:
            - Start with a top-level header (###) for the text title.
            - Use up to five header levels for organization.
            - Use hyphens (-) exclusively for bullet points.
            - Never use other symbols (e.g., •, ‣, or ■) or characters (+, *, etc.) for bullets.
            - Indent bullet points according to the hierarchy of the Markdown outline.
            - Always indent subheaders under headers.
            - Use consistent indentation for hierarchy and clarity.
            - Never usng an introduction sentence (e.g., 'Here is the...') before the outline. 
            - Use header levels effectively to organize content within the outline.
            - Only use bullets and headers for formatting. Do not use any other type of formatting. 
    """

    starting_prompt = prompt = f"""
        Craft a long outline reflecting the main points of a chapter using Markdown formatting. Adhere to these rules:
            - Under each header, thoroughly summarize chapter's topics, key terms, and themes in detail. 
            - Under the same headers, list pertinent questions raised by the chapter.
        The aim is to organize the chapter's essence into relevant, detailed bullet points and questions.
    """

    second_prompt = f"""
        Using the text provided in the chapter, increase the content of the outline while maintaining original Markdown formatting. In your responses, adhere to these rules:
            - Expand each bullet point with detailed explanations and insights based on the document's content.
            - Answer the questions posed in the outline.
            - When appropriate, define terms or concepts.
            - Use the chapter for evidence and context.
        Aim for detailed and informative responses that enhance understanding of the subject matter.

        Outline:
    """
    
    prompt_list = [
        {
            "prompt": starting_prompt, 
            "model_order": [{
                    "provider": "openai", 
                    "model": "4o-mini"
            }],
            "system_instructions": system_instructions
        },
        {
            "prompt": second_prompt, 
            "model_order": [{
                    "provider": "openai", 
                    "model": "4o-mini"
            }],
            "system_instructions": system_instructions
        },
    ]

    summary = summarize_with_revisions(
        text=chapter_text,
        prompt_list=prompt_list
    )

    return {
        'type': 'summary',
        'text': summary,
        'title': title,
        'chapter': chapter,
        'author': author,
        'publisher': publisher,
    }

if __name__ == '__main__':
    from genai_toolbox.helper_functions.string_helpers import retrieve_file
    from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding, find_similar_chunks
    import os
    import json
    from concurrent.futures import ThreadPoolExecutor

    

    summaries = []

    # with ThreadPoolExecutor() as executor:
    #     # Submit tasks for each text file
    #     futures = {
    #         executor.submit(
    #             summarize_file, 
    #             filename
    #         ): filename for filename in os.listdir(directory) if filename.endswith('.txt')
    #     }
        
    #     for future in futures:
    #         summary_entry = future.result()
    #         summaries.append(summary_entry)

    # with open('extracted_documents/all_summaries.json', 'w') as json_file:
    #     json.dump(summaries, json_file, indent=4)

   

    # from genai_toolbox.chunk_and_embed.embedding_functions import (
    #     create_openai_embedding, 
    #     embed_dict_list
    # )

    # summaries = retrieve_file('extracted_documents/all_summaries.json')
    # embedded_summaries = embed_dict_list(
    #     embedding_function=create_openai_embedding,
    #     chunk_dicts=summaries,
    #     model_choice = 'text-embedding-3-large',
    #     metadata_keys=[ 'chapter', 'title', 'author']
    # )
    # with open('extracted_documents/all_embedded_summaries.json', 'w') as json_file:
    #     json.dump(embedded_summaries, json_file, indent=4)


    with open('extracted_documents/all_embedded_summaries.json', 'r') as json_file:
        embedded_summaries = json.load(json_file)

    with open('extracted_documents/all_books_paragraphs.json', 'r') as json_file:
        all_books_paragraphs = json.load(json_file)
    
    all_books_paragraphs.extend(embedded_summaries)

    with open('extracted_documents/all_books_paragraphs.json', 'w') as json_file:
        json.dump(all_books_paragraphs, json_file, indent=4)