import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString
import os

book_1 = '../the-philosophical-baby-alison-gopnik-first-edition copy.epub'
book_2 = '../the-code-breaker-jennifer-doudna-gene-editing-and--annas-archive--libgenrs-nf-2933774 copy.epub'
book_3 = '../the-first-tycoon-the-epic-life-of-cornelius copy.epub'

# Open the EPUB file
book = epub.read_epub(book_1, options={"ignore_ncx": True})

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
        
        print(f"Extracted: {text_filename}\n{type(item)}")