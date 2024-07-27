from bs4 import BeautifulSoup
import re
import json
from num2words import num2words
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Any
import logging

load_dotenv()

from genai_toolbox.text_prompting.model_calls import openai_text_response
from genai_toolbox.helper_functions.string_helpers import evaluate_and_clean_valid_response

def extract_content(
    soup: BeautifulSoup
) -> Dict[str, Any]:
    content = []
    for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'img', 'span']):
        if element.name in ['h1', 'h2', 'h3']:
            content.append({'type': 'heading', 'level': int(element.name[1]), 'text': element.text.strip()})
        elif element.name == 'p':
            content.append({'type': 'paragraph', 'text': element.text.strip()})
        elif element.name == 'img':
            content.append({'type': 'image', 'src': element.get('src', ''), 'alt': element.get('alt', '')})
        elif element.name == 'span':
            content.append({'type': 'span', 'class': element.get('class', []), 'text': element.text.strip()})
    
    return {'content': content}

def create_hierarchy(
    content: Dict[str, Any]
) -> List[Dict[str, Any]]:
    hierarchy = []
    current_section = None
    current_subsection = None
    
    for item in content['content']:
        if item['type'] == 'heading' and item['level'] in [1, 2]:
            if current_section:
                hierarchy.append(current_section)
            current_section = {
                'type': 'section',
                'heading': item['text'],
                'content': []
            }
            current_subsection = None
        elif item['type'] == 'heading' and item['level'] == 3 and current_section:
            current_subsection = {'type': 'subsection', 'heading': item['text'], 'content': []}
            current_section['content'].append(current_subsection)
        else:
            if not current_section:
                current_section = {
                    'type': 'section',
                    'heading': None,
                    'content': []
                }
            # Add content directly to the section if there's no current subsection
            if current_subsection:
                current_subsection['content'].append(item)
            else:
                current_section['content'].append(item)
    
    if current_section:
        hierarchy.append(current_section)
    
    return hierarchy

def add_hierarchy_keys(
    hierarchy: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    def traverse(items, current_section=None, current_subsection=None):
        for item in items:
            if item['type'] == 'section':
                current_section = item['heading']
                item['section'] = current_section
                if 'content' in item:
                    traverse(item['content'], current_section)
            elif item['type'] == 'subsection':
                current_subsection = item['heading']
                item['section'] = current_section
                item['subsection'] = current_subsection
                if 'content' in item:
                    traverse(item['content'], current_section, current_subsection)
            else:
                item['section'] = current_section
                item['subsection'] = current_subsection
    
    traverse(hierarchy)
    return hierarchy

def extract_paragraphs(
    hierarchy: List[Dict[str, Any]],
    chapter: Optional[str] = None,
    title: Optional[str] = None
) -> List[Dict[str, Any]]:
    paragraphs = []

    def traverse(items):
        for item in items:
            if item['type'] == 'paragraph':
                item['chapter'] = chapter
                item['title'] = title
                paragraphs.append(item)
            elif 'content' in item:
                traverse(item['content'])

    traverse(hierarchy)
    return paragraphs

def identify_chapter_and_title(
    hierarchy: List[Dict[str, Any]],
    text: str,
) -> Tuple[Optional[str], Optional[str]]:
    word_to_num = _create_word_to_num_dict()
    word_pattern = '|'.join(re.escape(word) for word in word_to_num.keys())

    chapter, title = _find_chapter_and_title_from_hierarchy(hierarchy, word_to_num, word_pattern)
    logging.info(f"Found from hierarchy - chapter: {chapter}, title: {title}")

    if not chapter or not title:
        chapter, title = _get_chapter_and_title_from_ai(text, chapter, title)
        logging.info(f"Found from AI - chapter: {chapter}, title: {title}")
    
    logging.debug(f"Final result - chapter: {chapter}, title: {title}")
    return chapter, title

def _create_word_to_num_dict(
) -> Dict[str, str]:
    word_to_num = {num2words(i): str(i) for i in range(1, 101)}
    
    word_to_num.update({num2words(i).replace(' ', '-'): str(i) for i in range(21, 100)})
    
    for i in range(1, 4):
        base = i * 100
        word_to_num[num2words(base)] = str(base)
        word_to_num[f'{num2words(base)} and'] = str(base)
        
        for j in range(1, 100):
            num = base + j
            word = num2words(num).replace(' and ', ' ')
            word_to_num[word] = str(num)
            word_to_num[word.replace(' ', '-')] = str(num)
    return word_to_num

def _find_chapter_and_title_from_hierarchy(
    hierarchy: List[Dict[str, Any]],
    word_to_num: Dict[str, str],
    word_pattern: str
) -> Tuple[Optional[str], Optional[str]]:
    chapter = None
    title = None
    
    for item in hierarchy:
        if item['type'] == 'section':
            heading = item.get('heading')
            if heading is None:
                continue
            
            heading = re.sub(r'\s+', ' ', heading).strip()
            
            chapter_match = re.search(r'(?:C\s*H\s*A\s*P\s*T\s*E\s*R|CHAPTER|^)\s*(\d+|' + word_pattern + r')\.?', heading, re.IGNORECASE)
            if chapter_match:
                chapter_num = chapter_match.group(1).lower()
                chapter_num = word_to_num.get(chapter_num, chapter_num)
                chapter = f"Chapter {chapter_num}"
                
                title = re.sub(r'^(?:C\s*H\s*A\s*P\s*T\s*E\s*R|CHAPTER)?\s*(?:\d+|' + word_pattern + r')\.?\s*', '', heading, flags=re.IGNORECASE).strip()
                
                if chapter and title:
                    return chapter, title
            elif not title:
                title = heading
    
    return chapter, title

def _get_chapter_and_title_from_ai(
    text: str,
    current_chapter: Optional[str],
    current_title: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    first_10_lines = [line.strip() for line in text.split('\n') if line.strip()][:10]
    if not first_10_lines:
        logging.warning("No non-empty lines found in the first 10 lines of the text.")
        return current_chapter, current_title

    if len(first_10_lines) == 1:
        formatted_lines = first_10_lines[0]
    else:
        formatted_lines = "\n\n The first line: " + first_10_lines[0] + "\n\n The second line: " + first_10_lines[1] + "\n\n The rest of the lines:\n" + "\n".join(first_10_lines[2:])
    
    logging.info(f"Formatted lines: {formatted_lines}")

    prompt = f"""
    This is the first {'line' if len(first_10_lines) == 1 else '10 lines'} of a section from a book: 
    {formatted_lines}
    
    Based on the first few lines, what is the chapter number (if any) and title of this section? 
    The first line is probably the chapter (i.e. Chapter 1). 
    If the first line is not the chapter, it's probably the title.
    
    If there's no clear chapter number, return None.
    If there's no clear title, return None.
    
    If there's a list of Chapters, the Chapter is Table of Contents

    Return only JSON in a dictionary like this:
    {{
        "chapter": "Chapter 1",  // or None if no chapter number is found
        "title": "The Section Title"
    }},
    {{
        "chapter": None,
        "title": "The Section Title"
    }},
    {{
        "chapter": "Chapter 3",
        "title": None
    }},
    """
    try: 
        response = openai_text_response(prompt, model_choice="4o-mini")
        null_replaced = response.replace('null', 'None')
        result = evaluate_and_clean_valid_response(null_replaced, dict)
        logging.info(f"Response from OpenAI: {result}")
        
        if isinstance(result, dict):
            new_chapter = result.get('chapter')
            new_title = result.get('title')
            
            # Handle null values
            new_chapter = None if new_chapter is None else new_chapter
            new_title = None if new_title is None else new_title
            
            return (new_chapter if new_chapter is not None else current_chapter, 
                    new_title if new_title is not None else current_title)
        else:
            logging.debug(f"Unexpected response format: {response}")
    except json.JSONDecodeError:
        logging.debug(f"Failed to parse JSON response: {response}")
    except Exception as e:
        logging.error(f"Unexpected error processing AI response: {e}")
    
    return current_chapter, current_title

if __name__ == "__main__":
    import json
    import os

    hierarchy_file = 'extracted_documents/2_hierarchy.json'
    text_file = 'extracted_documents/2_document.txt'

    print(f"Size of {hierarchy_file}: {os.path.getsize(hierarchy_file) if os.path.exists(hierarchy_file) else 'File not found'} bytes")
    print(f"Size of {text_file}: {os.path.getsize(text_file) if os.path.exists(text_file) else 'File not found'} bytes")

    # Check if hierarchy file exists and has content
    if os.path.exists(hierarchy_file) and os.path.getsize(hierarchy_file) > 0:
        with open(hierarchy_file, 'r', encoding='utf-8') as file:
            try:
                hierarchy = json.load(file)
                print("Hierarchy loaded successfully.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {hierarchy_file}: {e}")
                print("File content:")
                with open(hierarchy_file, 'r', encoding='utf-8') as f:
                    print(f.read())
                hierarchy = None
    else:
        print(f"Hierarchy file {hierarchy_file} is empty or does not exist.")
        hierarchy = None

    # Check if text file exists and has content
    if os.path.exists(text_file) and os.path.getsize(text_file) > 0:
        with open(text_file, 'r', encoding='utf-8') as file:
            text = file.read()
        print("Text file loaded successfully.")
    else:
        print(f"Text file {text_file} is empty or does not exist.")
        text = None

    # Print loaded data
    if hierarchy:
        print("Hierarchy:", hierarchy)
    if text:
        print("Text:", text[:100] + "..." if len(text) > 100 else text)

    print(identify_chapter_and_title(hierarchy, text))