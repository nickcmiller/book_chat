from bs4 import BeautifulSoup, NavigableString, Tag
import re
import json
from num2words import num2words
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Any
import logging

load_dotenv()

from genai_toolbox.text_prompting.model_calls import openai_text_response
from genai_toolbox.helper_functions.string_helpers import evaluate_and_clean_valid_response

def extract_text_with_structure(
    element: NavigableString | Tag
) -> str:
    if isinstance(element, NavigableString):
        return str(element)
    elif element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'li']:
        return element.get_text(separator=' ', strip=False) + '\n\n'
    else:
        return ''.join(extract_text_with_structure(child) for child in element.children)

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
    author: Optional[str] = None,
    title: Optional[str] = None
) -> List[Dict[str, Any]]:
    paragraphs = []

    def traverse(items):
        for item in items:
            if item['type'] == 'paragraph':
                item['chapter'] = chapter
                item['author'] = author
                item['title'] = title
                paragraphs.append(item)
            elif 'content' in item:
                traverse(item['content'])

    traverse(hierarchy)
    return paragraphs

if __name__ == "__main__":
    import json
    import os