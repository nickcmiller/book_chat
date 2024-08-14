from typing import List, Dict, Any 
import tiktoken

from dotenv import load_dotenv
load_dotenv()

from genai_toolbox.text_prompting.model_calls import fallback_text_response

def summarize_with_revisions(
    text: str,
    prompt_list: List[Dict[str, Any]]
) -> str:
    """
        Summarizes the given text using a list of prompts and model orders.

        Parameters:
        - text (str): The text to be summarized.
        - prompt_list (List[Dict[str, Any]]): A list of dictionaries, each containing:
            - 'prompt' (str): The prompt to be used for summarization.
            - 'model_order' (List[Dict[str, str]]): The order of models to be used for generating the summary.
            - 'system_instructions' (str): Instructions for the model on how to generate the summary.

        Returns:
        - str: The final summary generated from the provided text using the specified prompts and models.
        
        The function iterates through the prompt list, creating a structured prompt for each entry and calling
        the `fallback_text_response` function to generate a summary. The context is updated with each generated
        summary to provide continuity in the summarization process.
    """
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
    """
        Constructs a structured prompt for the model by combining the provided prompt, context, and text.

        Parameters:
        - prompt (str): The initial prompt that guides the model's response.
        - context (str): The previous context or summary that provides continuity for the model.
        - text (str): The chapter text that the model will summarize or analyze.

        Returns:
        - str: A formatted string that combines the prompt, context, and chapter text, ready for model input.

        This function formats the input for the model by ensuring that the prompt, context, and text are clearly delineated,
        allowing the model to generate a coherent and contextually relevant response.
    """
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
    """
        Summarizes a chapter by creating a structured outline based on the chapter text and metadata.

        Parameters:
        - chapter_text (str): The text of the chapter to be summarized.
        - chapter_title (str): The title of the chapter being summarized.
        - metadata (Dict[str, Any]): A dictionary containing metadata about the chapter, including the author, title, and publisher.

        Returns:
        - str: A detailed outline of the chapter, formatted in Markdown, reflecting the main points, key terms, concepts, and themes, along with pertinent questions raised by the chapter.

        The function first constructs a starting prompt that instructs the model to create a long outline reflecting the main points of the chapter. It then defines a second prompt to expand the outline with detailed explanations and insights based on the chapter text. The system instructions ensure that the outline adheres to specific formatting rules for clarity and organization.
    """
    chapter = chapter_title
    author = metadata['creator']
    title = metadata['title']
    publisher = metadata['publisher']
    
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
            - Identify the chapter number and title in the first header.
            - Under each header, thoroughly summarize chapter's topics, key terms, concepts, and themes in detail. 
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

    if len(chapter_text) < 5000 or chapter_text is None:
        summary = ''
    else:
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