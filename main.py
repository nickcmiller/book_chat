import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString
import os
import json
from extraction_functions import extract_content, create_hierarchy, add_hierarchy_keys, extract_paragraphs, identify_chapter_and_title

import logging

logging.basicConfig(level=logging.INFO)

book_1 = '../the-philosophical-baby-alison-gopnik-first-edition copy.epub'
book_2 = '../the-code-breaker-jennifer-doudna-gene-editing-and--annas-archive--libgenrs-nf-2933774 copy.epub'
book_3 = '../the-first-tycoon-the-epic-life-of-cornelius copy.epub'
book_4 = '../Deep Utopia _ Life and Meaning in a Solved World -- Nick Bostrom -- 1, 2024 -- Ideapress Publishing copy.epub'

# Open the EPUB file
book = epub.read_epub(book_4, options={"ignore_ncx": True})

# Create a directory to store the extracted chapters
output_dir = 'extracted_documents'
os.makedirs(output_dir, exist_ok=True)

for i, item in enumerate(book.get_items()):
    # print(f"Item {i+1}: Type = {item.get_type()}")
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        # Process each document in the directory
        content = item.get_content()
        
        # Parse HTML content
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract text from HTML, preserving structure
        def extract_text_with_structure(element):
            if isinstance(element, NavigableString):
                return str(element)
            elif element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'li']:
                return element.get_text(separator=' ', strip=False) + '\n\n'
            else:
                return ''.join(extract_text_with_structure(child) for child in element.children)

        text = extract_text_with_structure(soup.body)
        html = soup.prettify()
        
        # Generate filenames for the chapter
        text_filename = f'{i+1}_document.txt'
        text_filepath = os.path.join(output_dir, text_filename)
        
        html_filename = f'{i+1}_document.html'
        html_filepath = os.path.join(output_dir, html_filename)
        
        # Save the extracted text to a file
        with open(text_filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        with open(html_filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        content = extract_content(soup)
        hierarchy = create_hierarchy(content)
        new_hierarchy = add_hierarchy_keys(hierarchy)
        chapter, title = identify_chapter_and_title(new_hierarchy, text)

        content_filename = f'{i+1}_content.json'
        content_filepath = os.path.join(output_dir, content_filename)
        
        hierarchy_filename = f'{i+1}_hierarchy.json'
        hierarchy_filepath = os.path.join(output_dir, hierarchy_filename)

        with open(content_filepath, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=4, ensure_ascii=False)

        with open(hierarchy_filepath, 'w', encoding='utf-8') as f:
            json.dump(new_hierarchy, f, indent=4, ensure_ascii=False)
        
        paragraphs = extract_paragraphs(new_hierarchy, chapter, title)
        if len(paragraphs) != 0 and (chapter is not None or title is not None):
            if chapter is not None and title is not None:
                paragraphs_filename = f'{i+1}_{chapter}_{title}_paragraphs.json'
            elif chapter is not None:
                paragraphs_filename = f'{i+1}_{chapter}_paragraphs.json'
            elif title is not None:
                paragraphs_filename = f'{i+1}_{title}_paragraphs.json'
            else:
                paragraphs_filename = f'{i+1}_paragraphs.json'
            
            paragraphs_filepath = os.path.join(output_dir, paragraphs_filename)

            with open(paragraphs_filepath, 'w', encoding='utf-8') as f:
                json.dump(paragraphs, f, indent=4, ensure_ascii=False)

        