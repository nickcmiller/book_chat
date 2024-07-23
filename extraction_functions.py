from bs4 import BeautifulSoup
import re
from num2words import num2words
import tiktoken
from dotenv import load_dotenv

load_dotenv()

from genai_toolbox.text_prompting.model_calls import openai_text_response
from genai_toolbox.helper_functions.string_helpers import evaluate_and_clean_valid_response

with open('extracted_documents/22_document.html', 'r') as file:
    html_content = file.read()

def parse_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup

def extract_content(soup):
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

def create_hierarchy(content):
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

def _create_word_to_num_dict():
    # Generate numbers 1-100 and their word representations
    word_to_num = {num2words(i): str(i) for i in range(1, 101)}
    
    # Add hyphenated versions for 21-99
    word_to_num.update({num2words(i).replace(' ', '-'): str(i) for i in range(21, 100)})
    
    # Add 'hundred' and variations for 100-300
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

def add_hierarchy_keys(hierarchy):
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

# def get_first_10_lines(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         return lines[:10]

# def clean_text(text):
#     # Remove HTML tags and special characters
#     clean = re.sub('<[^<]+?>', '', text)
#     clean = re.sub(r'\s+', ' ', clean).strip()
#     return clean

def identify_chapter_and_title(hierarchy):
    chapter = None
    title = None
    
    word_to_num = _create_word_to_num_dict()
    
    # Compile a regex pattern for matching word numbers
    word_pattern = r'\b(' + '|'.join(re.escape(key) for key in word_to_num.keys()) + r')\b'
    
    for item in hierarchy:
        if item['type'] == 'section':
            heading = item.get('heading')
            
            if heading is None:
                continue
            
            # Remove extra whitespace and newlines
            heading = re.sub(r'\s+', ' ', heading).strip()
            
            # Check for chapter number in the heading (including word numbers)
            chapter_match = re.search(r'(?:C\s*H\s*A\s*P\s*T\s*E\s*R|CHAPTER|^)\s*(\d+|' + word_pattern + r')\.?', heading, re.IGNORECASE)
            if chapter_match:
                chapter_num = chapter_match.group(1).lower()
                chapter_num = word_to_num.get(chapter_num, chapter_num)  # Convert word to number if necessary
                chapter = f"Chapter {chapter_num}"
                # If the entire heading is just the chapter number, continue to the next item
                if heading.lower() == chapter_match.group(0).strip().lower():
                    continue
            
            # If we haven't found a title yet, use this heading as the title
            if not title:
                # Remove the chapter number from the heading if it exists
                title = re.sub(r'^(?:C\s*H\s*A\s*P\s*T\s*E\s*R|CHAPTER)?\s*(?:\d+|' + word_pattern + r')\.?\s*', '', heading, flags=re.IGNORECASE).strip()
            
            # If we've found both chapter and title, break the loop
            if chapter and title:
                break
    
    # If we still don't have a title, check the content for a span with class 'heading_break1'
    def extract_text_with_structure(element):
        if isinstance(element, NavigableString):
            return str(element)
        elif element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'li']:
            return element.get_text(separator=' ', strip=False) + '\n\n'
        else:
            return ''.join(extract_text_with_structure(child) for child in element.children)

    text = extract_text_with_structure(soup.body)
    first_10_lines = get_first_10_lines('extracted_documents/22_document.txt')
    prompt = f"""
    This is the first 10 lines of the chapter: {first_10_lines}.
    
    What is the chapter and title of this chapter?

    Return only JSON in a dictionary like this:
    {{
        "chapter": "Chapter 1",
        "title": "The First Chapter"
    }}
    """
    response = openai_text_response(prompt, model_choice="4o-mini")
    print(response)
    print(evaluate_and_clean_valid_response(response, dict))
    
    
    return chapter, title

if __name__ == "__main__":
    import json

    soup = parse_html(html_content)
    content = extract_content(soup)
    hierarchy = create_hierarchy(content)
    new_hierarchy = add_hierarchy_keys(hierarchy)
    
    with open('content.json', 'w') as file:
        json.dump(content, file, indent=4, ensure_ascii=False)

    with open('hierarchy.json', 'w') as file:
        json.dump(new_hierarchy, file, indent=4, ensure_ascii=False)