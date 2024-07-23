from bs4 import BeautifulSoup
import re
from num2words import num2words

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

def create_word_to_num_dict():
    word_to_num = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90', 'hundred': '100'
    }

    # Add compound numbers
    for tens in ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']:
        for ones in range(1, 10):
            compound = f"{tens}-{num2words(ones)}"
            word_to_num[compound] = str(int(word_to_num[tens]) + ones)
            compound = f"{tens} {num2words(ones)}"
            word_to_num[compound] = str(int(word_to_num[tens]) + ones)

    # Add numbers 100-300
    for i in range(1, 4):
        hundreds = num2words(i * 100)
        word_to_num[hundreds] = str(i * 100)
        word_to_num[f'{hundreds} and'] = str(i * 100)
        for j in range(1, 100):
            word = num2words(i * 100 + j).replace(' and ', ' ')
            word_to_num[word] = str(i * 100 + j)
            word_to_num[word.replace(' ', '-')] = str(i * 100 + j)

    return word_to_num

def identify_chapter_and_title(hierarchy):
    chapter = None
    title = None
    
    word_to_num = create_word_to_num_dict()
    
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
    if not title:
        for item in hierarchy:
            if 'content' in item:
                for content_item in item['content']:
                    if content_item.get('type') == 'span' and 'heading_break1' in content_item.get('class', []):
                        title = content_item.get('text', '').strip()
                        if title:
                            break
                if title:
                    break
    
    return chapter, title

def add_hierarchy_keys(hierarchy, chapter, title):
    def traverse(items, current_section=None, current_subsection=None):
        for item in items:
            item['chapter'] = chapter
            item['title'] = title
            
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

def clean_text(text):
    # Remove HTML tags and special characters
    clean = re.sub('<[^<]+?>', '', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

if __name__ == "__main__":
    import json

    soup = parse_html(html_content)
    content = extract_content(soup)
    hierarchy = create_hierarchy(content)
    chapter, title = identify_chapter_and_title(hierarchy)
    print(f"Chapter: {chapter}")
    print(f"Title: {title}")
    # new_hierarchy = add_hierarchy_keys(hierarchy, chapter, title)
    

    with open('content.json', 'w') as file:
        json.dump(content, file, indent=4, ensure_ascii=False)

    with open('hierarchy.json', 'w') as file:
        json.dump(hierarchy, file, indent=4, ensure_ascii=False)