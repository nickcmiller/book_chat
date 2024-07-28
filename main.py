import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString, Tag
import os
import json
import re
from extraction_functions import (
    extract_text_with_structure, 
    extract_content, 
    create_hierarchy, 
    add_hierarchy_keys, 
    extract_paragraphs, 
    safe_write_file, 
    create_toc_mapping, 
    toc_to_text, 
    eliminate_fragments, 
    extract_metadata
)
import logging

logging.basicConfig(level=logging.INFO)

book_1 = '../the-philosophical-baby-alison-gopnik-first-edition copy.epub'
book_2 = '../the-code-breaker-jennifer-doudna-gene-editing-and--annas-archive--libgenrs-nf-2933774 copy.epub'
book_3 = '../the-first-tycoon-the-epic-life-of-cornelius copy.epub'
book_4 = '../Deep Utopia _ Life and Meaning in a Solved World -- Nick Bostrom -- 1, 2024 -- Ideapress Publishing copy.epub'
book_path = book_3
# Open the EPUB file

book = epub.read_epub(book_path)
toc_text = toc_to_text(book)
toc_mapping = create_toc_mapping(book)
chapter_mapping = eliminate_fragments(toc_mapping)
print(f"\n\ntoc_text: {toc_text}\n\n")
print(f"\n\nchapter_mapping: {json.dumps(chapter_mapping, indent=2)}\n\n")

metadata = extract_metadata(book)
print(f"\n\nmetadata: {json.dumps(metadata, indent=2)}\n\n")

# Create a directory to store the extracted chapters
output_dir = 'extracted_documents'
os.makedirs(output_dir, exist_ok=True)

for i, item in enumerate(book.get_items()):
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        file_name = item.get_name()
        
        chapter_title = chapter_mapping.get(file_name)
        print(f"\n\nfile_name: {file_name} \nchapter_title: {chapter_title}\n\n")
        
        if chapter_title is None:
            print(f"Skipping {file_name}: No mapped chapter title found")
            continue
        
        safe_chapter_title = re.sub(r'[^\w\-_\. ]', '_', chapter_title).replace(' ', '_')
        print(f"{file_name}: {safe_chapter_title}")

        soup = BeautifulSoup(item.get_content(), 'html.parser')

        text = extract_text_with_structure(soup.body)
        html = soup.prettify()
        content = extract_content(soup)
        hierarchy = create_hierarchy(content)
        new_hierarchy = add_hierarchy_keys(hierarchy)
        paragraphs = extract_paragraphs(new_hierarchy, safe_chapter_title, metadata['creator'], metadata['title'])

        text_filename = f"{i}_{safe_chapter_title}.txt"
        text_filepath = os.path.join(output_dir, text_filename)
        safe_write_file(text, text_filepath, file_type='text')

        hierarchy_filename = f'{i+1}_{safe_chapter_title}_hierarchy.json'
        hierarchy_filepath = os.path.join(output_dir, hierarchy_filename)
        safe_write_file(new_hierarchy, hierarchy_filepath)

        paragraphs_filename = f'{i+1}_{safe_chapter_title}_paragraphs.json'
        paragraphs_filepath = os.path.join(output_dir, paragraphs_filename)
        safe_write_file(paragraphs, paragraphs_filepath)