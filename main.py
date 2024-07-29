import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
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
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)

EXTRACTED_DIR = 'extracted_documents'

def setup_output_directory(
    book_name: str
) -> str:
    output_dir = os.path.join(EXTRACTED_DIR, book_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def process_chapter(
    item: epub.EpubItem,
    chapter_mapping: dict,
    metadata: dict,
    output_dir: str,
    i: int
) -> List[Dict[str, Any]]:
    file_name = item.get_name()
    chapter_title = chapter_mapping.get(file_name)
    
    if chapter_title is None:
        logging.info(f"Skipping {file_name}: No mapped chapter title found")
        return []
    
    safe_chapter_title = re.sub(r'[^\w\-_\. ]', '_', chapter_title).replace(' ', '_')
    logging.info(f"{file_name}: {safe_chapter_title}")

    soup = BeautifulSoup(item.get_content(), 'html.parser')

    text = extract_text_with_structure(soup.body)
    content = extract_content(soup)
    hierarchy = create_hierarchy(content)
    new_hierarchy = add_hierarchy_keys(hierarchy)
    paragraphs = extract_paragraphs(new_hierarchy, safe_chapter_title, metadata['creator'], metadata['title'])

    save_chapter_files(text, new_hierarchy, paragraphs, output_dir, i, safe_chapter_title)

    return paragraphs

def save_chapter_files(
    text: str,
    hierarchy: dict,
    paragraphs: List[Dict[str, Any]],
    output_dir: str,
    i: int,
    safe_chapter_title: str
) -> None:
    text_filename = f"{i}_{safe_chapter_title}.txt"
    text_filepath = os.path.join(output_dir, text_filename)
    safe_write_file(text, text_filepath, file_type='text')

    hierarchy_filename = f'{i+1}_{safe_chapter_title}_hierarchy.json'
    hierarchy_filepath = os.path.join(output_dir, hierarchy_filename)
    safe_write_file(hierarchy, hierarchy_filepath)

    paragraphs_filename = f'{i+1}_{safe_chapter_title}_paragraphs.json'
    paragraphs_filepath = os.path.join(output_dir, paragraphs_filename)
    safe_write_file(paragraphs, paragraphs_filepath)

def process_book(
    book_path: str
) -> None:
    logging.info(f"Processing book: {book_path}")
    book = epub.read_epub(book_path)
    
    toc_mapping = create_toc_mapping(book)
    chapter_mapping = {**toc_mapping, **eliminate_fragments(toc_mapping)}
    toc_text = toc_to_text(book)
    metadata = extract_metadata(book)

    logging.info(f"TOC text: {toc_text}")
    logging.info(f"Chapter mapping: {json.dumps(chapter_mapping, indent=2)}")
    logging.info(f"Metadata: {json.dumps(metadata, indent=2)}")

    book_name = metadata['title'].replace(' ', '_')
    output_dir = setup_output_directory(book_name)

    all_paragraphs = []

    for i, item in enumerate(book.get_items()):
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            paragraphs = process_chapter(item, chapter_mapping, metadata, output_dir, i)
            all_paragraphs.extend(paragraphs)

    save_consolidated_paragraphs(all_paragraphs, book_name, output_dir)
    logging.info(f"Finished processing book: {book_path}")

def save_consolidated_paragraphs(
    all_paragraphs: List[Dict[str, Any]],
    book_name: str,
    output_dir: str
) -> None:
    print(f"Number of paragraphs: {len(all_paragraphs)}")
    consolidated_filename = f'{book_name}_all_paragraphs.json'
    consolidated_filepath = os.path.join(output_dir, consolidated_filename)
    safe_write_file(all_paragraphs, f"{EXTRACTED_DIR}/{book_name}_all_paragraphs.json")
    logging.info(f"Consolidated paragraphs saved to: {consolidated_filepath}")

def process_books(
    book_paths: List[str]
) -> None:
    for book_path in book_paths:
        process_book(book_path)

if __name__ == "__main__":
    book_paths = [
        '../the-philosophical-baby-alison-gopnik-first-edition copy.epub',
        '../the-code-breaker-jennifer-doudna-gene-editing.epub',
        '../the-first-tycoon-the-epic-life-of-cornelius copy.epub',
        '../Deep Utopia _ Life and Meaning in a Solved World -- Nick Bostrom -- 1, 2024 -- Ideapress Publishing copy.epub'
    ]
    process_books(book_paths)