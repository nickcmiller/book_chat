from bs4 import BeautifulSoup, NavigableString, Tag
import re
import json
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Any
import logging
import os
import ebooklib

load_dotenv()

def extract_metadata(book):
    metadata = {}
    metadata['title'] = book.get_metadata('DC', 'title')
    metadata['creator'] = book.get_metadata('DC', 'creator')
    metadata['language'] = book.get_metadata('DC', 'language')
    metadata['identifier'] = book.get_metadata('DC', 'identifier')
    metadata['publisher'] = book.get_metadata('DC', 'publisher')
    metadata['date'] = book.get_metadata('DC', 'date')
    return {k: v[0][0] if v else None for k, v in metadata.items()}

def create_toc_mapping(book):
    ncx_item = next((item for item in book.get_items() if item.get_type() == ebooklib.ITEM_NAVIGATION), None)
    if not ncx_item:
        return None

    ncx_content = ncx_item.get_content().decode('utf-8')
    soup = BeautifulSoup(ncx_content, 'xml')
    nav_points = soup.find_all('navPoint')

    toc_mapping = {}

    def process_nav_point(nav_point):
        label = nav_point.navLabel.text.strip()
        content = nav_point.content['src']
        toc_mapping[content] = label  # Swap the key and value
        for child in nav_point.find_all('navPoint', recursive=False):
            process_nav_point(child)

    for nav_point in nav_points:
        process_nav_point(nav_point)

    return toc_mapping

def eliminate_fragments(toc_mapping):
    chapter_mapping = {}
    for file, title in toc_mapping.items():
        base_file = file.split('#')[0]
        if base_file not in chapter_mapping:
            chapter_mapping[base_file] = title
    return chapter_mapping

def toc_to_text(book):
    output = []
    output.append(f"Book Title: {book.get_metadata('DC', 'title')[0][0]}")
    output.append(f"Author: {book.get_metadata('DC', 'creator')[0][0]}")
    
    toc_mapping = create_toc_mapping(book)
    
    if toc_mapping:
        output.append("\nTable of Contents:")
        current_chapter = None
        for title, file in toc_mapping.items():
            if '#' not in file:
                current_chapter = title
                output.append(f"{title}")
            elif current_chapter:
                output.append(f"  - {title}")
            else:
                output.append(f"{title}")
    else:
        output.append("\nNo NCX file found or unable to create mapping.")
    
    return "\n".join(output)

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

def safe_write_file(content, file_path, file_type='json'):
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path

    while os.path.exists(new_file_path):
        new_file_path = f"{base}_{counter}{ext}"
        counter += 1

    with open(new_file_path, 'w', encoding='utf-8') as f:
        if file_type == 'json':
            json.dump(content, f, indent=4, ensure_ascii=False)
        else:
            f.write(content)

    return new_file_path

if __name__ == "__main__":
    import json
    import os