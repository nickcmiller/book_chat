import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString, Tag
import os
import json
import re
from extraction_functions import extract_text_with_structure, extract_content, create_hierarchy, add_hierarchy_keys, extract_paragraphs, identify_chapter_and_title
import logging

logging.basicConfig(level=logging.INFO)

book_1 = '../the-philosophical-baby-alison-gopnik-first-edition copy.epub'
book_2 = '../the-code-breaker-jennifer-doudna-gene-editing-and--annas-archive--libgenrs-nf-2933774 copy.epub'
book_3 = '../the-first-tycoon-the-epic-life-of-cornelius copy.epub'
book_4 = '../Deep Utopia _ Life and Meaning in a Solved World -- Nick Bostrom -- 1, 2024 -- Ideapress Publishing copy.epub'
book_to_process = book_1
# Open the EPUB file

def extract_toc(epub_path):
    book = epub.read_epub(epub_path)
    toc = book.toc
    
    def process_toc(toc_items, level=0):
        result = []
        for item in toc_items:
            if isinstance(item, tuple):
                section = {
                    'title': item[0].title,
                    'level': level,
                    'href': item[0].href,
                }
                result.append(section)
                if len(item) > 1 and isinstance(item[1], list):
                    result.extend(process_toc(item[1], level + 1))
            elif isinstance(item, epub.Link):
                result.append({
                    'title': item.title,
                    'level': level,
                    'href': item.href,
                })
        return result

    return process_toc(toc)

def toc_to_text(toc):
    return ''.join(f"{'  ' * item['level']}{item['title']}\n" for item in toc)

book_to_process = book_2
# After extracting the TOC, print it
toc = extract_toc(book_to_process)
print(toc_to_text(toc))

def extract_metadata(book):
    metadata = {}
    metadata['title'] = book.get_metadata('DC', 'title')
    metadata['creator'] = book.get_metadata('DC', 'creator')
    metadata['language'] = book.get_metadata('DC', 'language')
    metadata['identifier'] = book.get_metadata('DC', 'identifier')
    metadata['publisher'] = book.get_metadata('DC', 'publisher')
    metadata['date'] = book.get_metadata('DC', 'date')
    return {k: v[0][0] if v else None for k, v in metadata.items()}


# Create a mapping of hrefs to chapter titles
def build_href_mapping(toc_items):
    href_to_chapter = {}
    for item in toc_items:
        href = item['href'].split('#')[0]
        href_to_chapter[href] = item['title']
    return href_to_chapter


book = epub.read_epub(book_to_process, options={"ignore_ncx": True})
toc = extract_toc(book_to_process)
toc_text = toc_to_text(toc)
print(f"\n\ntoc_text: {toc_text}\n\n")
href_to_chapter = build_href_mapping(toc)
print(f"\n\nhref_to_chapter: {json.dumps(href_to_chapter, indent=2)}\n\n")
metadata = extract_metadata(book)
print(f"\n\nmetadata: {json.dumps(metadata, indent=2)}\n\n")

# Create a directory to store the extracted chapters
output_dir = 'extracted_documents'
os.makedirs(output_dir, exist_ok=True)

for i, item in enumerate(book.get_items()):
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        file_name = item.get_name()
        chapter_title = href_to_chapter.get(file_name, f"Chapter_{i+1}")
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
        with open(text_filepath, 'w', encoding='utf-8') as f:
            f.write(text)

        # html_filename = f"{i}_{safe_chapter_title}.html"
        # html_filepath = os.path.join(output_dir, html_filename)
        # with open(html_filepath, 'w', encoding='utf-8') as f:
        #     f.write(html)

        # content_filename = f'{i+1}_{safe_chapter_title}_content.json'
        # content_filepath = os.path.join(output_dir, content_filename)
        # with open(content_filepath, 'w', encoding='utf-8') as f:
        #     json.dump(content, f, indent=4, ensure_ascii=False)

        hierarchy_filename = f'{i+1}_{safe_chapter_title}_hierarchy.json'
        hierarchy_filepath = os.path.join(output_dir, hierarchy_filename)
        with open(hierarchy_filepath, 'w', encoding='utf-8') as f:
            json.dump(new_hierarchy, f, indent=4, ensure_ascii=False)

        paragraphs_filename = f'{i+1}_{safe_chapter_title}_paragraphs.json'
        paragraphs_filepath = os.path.join(output_dir, paragraphs_filename)
        with open(paragraphs_filepath, 'w', encoding='utf-8') as f:
            json.dump(paragraphs, f, indent=4, ensure_ascii=False)