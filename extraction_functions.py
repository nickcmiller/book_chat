from bs4 import BeautifulSoup, NavigableString, Tag
import ebooklib
from ebooklib import epub
from dotenv import load_dotenv
load_dotenv()

from genai_toolbox.chunk_and_embed.embedding_functions import num_tokens_from_string

from typing import List, Dict, Any, Optional 
import re
import json
from typing import List, Dict, Tuple, Optional, Any
import logging
import os

def extract_metadata(
    book: epub.EpubBook
) -> Dict[str, Any]:
    metadata = {}
    metadata['title'] = book.get_metadata('DC', 'title')
    metadata['creator'] = book.get_metadata('DC', 'creator')
    metadata['language'] = book.get_metadata('DC', 'language')
    metadata['identifier'] = book.get_metadata('DC', 'identifier')
    metadata['publisher'] = book.get_metadata('DC', 'publisher')
    metadata['date'] = book.get_metadata('DC', 'date')
    return {k: v[0][0] if v else None for k, v in metadata.items()}

def create_toc_mapping(
    book: epub.EpubBook
) -> Optional[Dict[str, str]]:
    """
        Create a mapping of the table of contents (TOC) for the given EPUB book.

        This function extracts the navigation points from the EPUB book's NCX (Navigation Control file for XML) 
        and creates a dictionary mapping the content source (file path) to the corresponding chapter title.

        Args:
            book (epub.EpubBook): The EPUB book object from which to extract the TOC.

        Returns:
            Optional[Dict[str, str]]: A dictionary mapping file paths to chapter titles, or None if no navigation 
            item is found in the book.
    """
    ncx_item = next((item for item in book.get_items() if item.get_type() == ebooklib.ITEM_NAVIGATION), None)
    if not ncx_item:
        return None

    ncx_content = ncx_item.get_content().decode('utf-8')
    soup = BeautifulSoup(ncx_content, 'xml')
    nav_points = soup.find_all('navPoint')

    toc_mapping = {}

    def process_nav_point(
        nav_point: Tag
    ) -> None:
        """
            Process a navigation point in the table of contents (TOC) and update the toc_mapping.

            This recursive function extracts the chapter title and content source from a given 
            navigation point (navPoint) in the EPUB book's NCX file. It adds the mapping of the 
            content source to the chapter title in the toc_mapping dictionary. If the navPoint 
            contains child navPoints, the function is called recursively to process them as well.

            Args:
                nav_point (BeautifulSoup): The navigation point element to process.
        """
        label = nav_point.navLabel.text.strip()
        content = nav_point.content['src']
        toc_mapping[content] = label
        for child in nav_point.find_all('navPoint', recursive=False):
            process_nav_point(child)

    for nav_point in nav_points:
        process_nav_point(nav_point)

    return toc_mapping

def eliminate_fragments(
    toc_mapping: Dict[str, str]
) -> Dict[str, str]:
    """
        Eliminate fragments from the table of contents mapping.

        This function takes a dictionary mapping file paths to chapter titles and removes any entries that are 
        considered fragments. A fragment is defined as a part of a file that is not a complete chapter, typically 
        indicated by the presence of a '#' in the file path. The function returns a new dictionary that maps 
        base file paths to their corresponding chapter titles.

        Args:
            toc_mapping (Dict[str, str]): A dictionary mapping file paths to chapter titles.

        Returns:
            Dict[str, str]: A new dictionary mapping base file paths to chapter titles, excluding fragments.
    """
    chapter_mapping = {}
    for file, title in toc_mapping.items():
        base_file = file.split('#')[0]
        if base_file not in chapter_mapping:
            chapter_mapping[base_file] = title
    return chapter_mapping

def toc_to_text(
    book: epub.EpubBook
) -> str:
    """
        Convert the table of contents (TOC) of the EPUB book into a text format.

        This function extracts the title and author from the book's metadata and generates a 
        formatted string representation of the table of contents. It organizes the chapters 
        and sub-chapters based on the mapping created from the NCX file, ensuring that 
        fragments (indicated by a '#' in the file path) are properly nested under their 
        respective chapters.

        Args:
            book (epub.EpubBook): The EPUB book object from which to extract the TOC.

        Returns:
            str: A formatted string representing the book's title, author, and table of contents.
    """
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
    """
        Extracts text from a BeautifulSoup element while preserving the structure of the document.

        This function recursively processes the given HTML element (or its children) to extract text 
        content. It handles different types of elements, including paragraphs, headings, and lists, 
        and formats the output accordingly. The function ensures that the text is returned in a 
        structured format, with appropriate line breaks for readability.

        Args:
            element (NavigableString | Tag): The BeautifulSoup element from which to extract text.

        Returns:
            str: The extracted text content, formatted with line breaks for structure.
    """
    if isinstance(element, NavigableString):
        return str(element)
    elif element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'li']:
        return element.get_text(separator=' ', strip=False) + '\n\n'
    else:
        return ''.join(extract_text_with_structure(child) for child in element.children)

def extract_content(
    soup: BeautifulSoup
) -> Dict[str, Any]:
    """
        Extracts content from a BeautifulSoup object and organizes it into a structured format.

        This function scans the provided BeautifulSoup object for specific HTML elements 
        (headings, paragraphs, images, and spans) and compiles their text and attributes 
        into a list of dictionaries. Each dictionary represents a piece of content with 
        its type and relevant information, allowing for easy processing and manipulation 
        of the extracted data.

        Args:
            soup (BeautifulSoup): The BeautifulSoup object representing the parsed HTML 
                                document from which to extract content.

        Returns:
            Dict[str, Any]: A dictionary containing a list of content items, where each 
                            item is a dictionary with keys 'type' and associated data.
    """
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
    """
        Creates a hierarchical structure from the extracted content, organizing it into sections and subsections.

        This function processes a dictionary containing content items (headings and paragraphs) and organizes them 
        into a nested list structure. Sections are created for top-level headings (h1 and h2), and subsections are 
        created for lower-level headings (h3). Each section and subsection contains its respective content, allowing 
        for a clear representation of the document's structure.

        Args:
            content (Dict[str, Any]): A dictionary containing a list of content items extracted from a BeautifulSoup 
                                       object, where each item has a 'type' and associated data.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the hierarchical structure of the content, 
                                  where each dictionary can represent a section or subsection with its content.
    """
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
    """
        Adds hierarchical keys to the sections and subsections in the provided hierarchy.

        This function traverses the hierarchy of sections and subsections, assigning 
        'section' and 'subsection' keys to each item. This allows for easier access 
        to the section and subsection information for each content item.

        Args:
            hierarchy (
                List[Dict[str, Any]]
            ): A list of dictionaries representing the hierarchical structure of the content, 
                where each dictionary can represent a section or subsection with its content.

        Returns:
            List[Dict[str, Any]]: The modified hierarchy with added 'section' and 
                                'subsection' keys for each item.
    """
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
    publisher: Optional[str] = None,
    title: Optional[str] = None,
    min_paragraph_tokens: int = 15
) -> List[Dict[str, Any]]:
    """
        Extracts paragraphs from the provided hierarchical structure.

        This function traverses the hierarchy of sections and subsections, 
        extracting paragraphs and associating them with their respective chapter, 
        author, and title information. Each paragraph is represented as a dictionary 
        containing its text and metadata.

        Args:
            hierarchy (List[Dict[str, Any]]): A list of dictionaries representing 
                the hierarchical structure of the content, where each dictionary 
                can represent a section, subsection, or paragraph.
            chapter (Optional[str]): The title of the chapter from which the 
                paragraphs are extracted. Defaults to None.
            author (Optional[str]): The author of the content. Defaults to None.
            title (Optional[str]): The title of the content. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a 
                paragraph with its text and associated metadata (chapter, author, 
                title, section, and subsection).
    """
    paragraphs = []

    def traverse(
        items: List[Dict[str, Any]]
    ) -> None:
        """
            Traverses the hierarchical structure of sections and subsections to extract 
            paragraphs along with their associated metadata.

            This inner function iterates through each item in the provided list of 
            dictionaries, checking the type of each item. If the item is a paragraph, 
            it creates a new dictionary containing the paragraph's text and its 
            associated metadata (chapter, author, title, section, and subsection) 
            and appends it to the paragraphs list. If the item contains further 
            content, it recursively calls itself to process that content.

            Args:
                items (List[Dict[str, Any]]): A list of dictionaries representing 
                    sections, subsections, or paragraphs in the hierarchy.
        """
        for item in items:
            if item['type'] == 'paragraph':
                new_paragraph = {
                    'type': 'paragraph',
                    'text': item['text'],
                    'title': title,
                    'chapter': chapter,
                    'author': author,
                    'publisher': publisher,
                    'section': item.get('section'),
                    'subsection': item.get('subsection')
                }
                paragraphs.append(new_paragraph)
            elif 'content' in item:
                traverse(item['content'])

    traverse(hierarchy)

    filtered_paragraphs = [p for p in paragraphs if num_tokens_from_string(p['text']) >= min_paragraph_tokens]

    return filtered_paragraphs

def safe_write_file(
    content: Any, 
    file_path: str, 
    file_type: str = 'json'
) -> str:
    """
        Safely writes content to a specified file path, ensuring that if a file 
        with the same name already exists, a new file name is generated by 
        appending a counter to the base name.

        Args:
            content (Any): The content to be written to the file. This can be 
                any data type, such as a dictionary (for JSON) or a string.
            file_path (str): The path where the file will be saved, including 
                the desired file name and extension.
            file_type (str, optional): The type of file to be created. Defaults 
                to 'json'. If 'json', the content will be serialized to JSON 
                format; otherwise, it will be written as plain text.

        Returns:
            str: The path of the newly created file, which may include a counter 
            if a file with the original name already existed.
    """
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

    from genai_toolbox.text_prompting.model_calls import perplexity_text_response

    history_messages = [
        {"role": "system", "content": "Write in the voice and style of Ozzy Osbourne"},
        {"role": "user", "content": "Who is David Nola at NVIDIA?"},
        {"role": "assistant", "content": "David Nola is a Deep Learning Solutions Architect at NVIDIA, based in Santa Clara, California. He has a strong educational background, having earned a Bachelor of Science in Finance and Computer Science (Double Major) from Santa Clara University and a Master of Science in Computer Science from the University of California, Los Angeles (UCLA). His professional experience includes various roles at NVIDIA, such as Deep Learning Solutions Architect Intern and Release Engineer at FileMaker. He has also held positions at other companies, including a Financial Analyst Intern at Amussen, Hunsaker & Associates Inc. and a Junior Data Analyst at Sorenson Communications. David Nola has been recognized for his academic achievements, including being on the Dean's List and receiving the Paul R. Halmos Prize in Mathematics from Santa Clara University."},
    ]

    response = perplexity_text_response(
        prompt="Write a hello world function",
        system_instructions="Be detailed and thorough",
        history_messages=history_messages,
        model_choice="llama3.1-8b",
    )
    print(response)