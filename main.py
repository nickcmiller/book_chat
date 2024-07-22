import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import os

# Open the EPUB file
book = epub.read_epub('../the-philosophical-baby-alison-gopnik-first-edition Copy.epub', options={"ignore_ncx": True})
print(type(book))

# Create a directory to store the extracted chapters
output_dir = 'extracted_chapters'
os.makedirs(output_dir, exist_ok=True)

# Iterate through the chapters
for i, item in enumerate(book.get_items()):
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        # Process the chapter content
        content = item.get_content()
        
        # Parse HTML content
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract text from HTML
        text = soup.get_text()
        
        # Generate a filename for the chapter
        filename = f'chapter_{i+1}.txt'
        filepath = os.path.join(output_dir, filename)
        
        # Save the extracted text to a file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Extracted: {filename}")