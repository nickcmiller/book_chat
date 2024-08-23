import os
import json
import copy
import re
import logging
from typing import List, Dict, Any
import concurrent.futures
from functools import partial
from collections import defaultdict

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

from extraction_functions import (
    extract_text_with_structure, 
    extract_content, 
    create_hierarchy, 
    add_hierarchy_keys, 
    extract_paragraphs, 
    safe_write_file, 
    create_toc_mapping, 
    decode_chapter_mapping,
    toc_to_text, 
    eliminate_fragments, 
    extract_metadata
)
from summarize_text import summarize_chapter
from genai_toolbox.chunk_and_embed.embedding_functions import (
    create_openai_embedding, 
    embed_dict_list
)
from genai_toolbox.helper_functions.string_helpers import (
    retrieve_file,
    write_to_file
)

logging.basicConfig(level=logging.INFO)

EXTRACTED_DIR = 'extracted_documents'

def setup_output_directory(
    book_name: str
) -> str:
    """
        Sets up an output directory for the book based on its name.

        Parameters:
        - book_name (str): The name of the book.

        Returns:
        - str: The path to the output directory.
    """
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
    """
        Processes a chapter from an EPUB item, extracting its text, content, and hierarchical structure.

        Parameters:
        - item (epub.EpubItem): The EPUB item representing the chapter to be processed.
        - chapter_mapping (dict): A mapping of file names to chapter titles.
        - metadata (dict): Metadata information about the book, including creator and title.
        - output_dir (str): The directory where the processed chapter files will be saved.
        - i (int): The index of the chapter being processed.

        Returns:
        - List[Dict[str, Any]]: A list of paragraphs extracted from the chapter, each represented as a dictionary.
        
        This function logs the processing of the chapter, extracts the text and content using BeautifulSoup,
        creates a hierarchy of the content, and saves the chapter's text, hierarchy, and paragraphs to files.
        If no mapped chapter title is found, it logs a message and skips processing for that chapter.
    """
    # print(f"\n\nchapter_mapping: {json.dumps(chapter_mapping, indent=2)}")
    file_name = item.get_name()
    chapter_title = chapter_mapping.get(file_name)
    
    if chapter_title is None:
        logging.info(f"Skipping {file_name}: No mapped chapter title found")
        return []
    
    safe_chapter_title = re.sub(r'[^\w\-_\. ]', '_', chapter_title).replace(' ', '_')
    logging.info(f"{file_name}: {safe_chapter_title}")

    soup = BeautifulSoup(item.get_content(), 'html.parser')

    text = extract_text_with_structure(soup.body)
    summary = summarize_chapter(text, chapter_title, metadata)
    content = extract_content(soup)
    hierarchy = create_hierarchy(content)
    new_hierarchy = add_hierarchy_keys(hierarchy)
    paragraphs = extract_paragraphs(
        hierarchy=new_hierarchy, 
        chapter=chapter_title, 
        author=metadata['creator'], 
        title=metadata['title'],
        publisher=metadata['publisher'],
        min_paragraph_tokens=15
    )
    paragraphs.append(summary)
    
    summary_text = summary['text'] if summary is not None and summary['text'] != '' else None

    _save_chapter_files(text, summary_text, new_hierarchy, paragraphs, output_dir, i, safe_chapter_title)

    return paragraphs
    
def _save_chapter_files(
    text: str,
    summary_text: str,
    hierarchy: dict,
    paragraphs: List[Dict[str, Any]],
    output_dir: str,
    i: int,
    safe_chapter_title: str
) -> None:
    """
        Saves the chapter files including the text, hierarchy, and paragraphs to the specified output directory.

        Parameters:
        - text (str): The text content of the chapter.
        - hierarchy (dict): The hierarchical structure of the chapter's content.
        - paragraphs (List[Dict[str, Any]]): A list of paragraphs extracted from the chapter.
        - output_dir (str): The directory where the chapter files will be saved.
        - i (int): The index of the chapter being processed.
        - safe_chapter_title (str): A safe version of the chapter title for use in filenames.

        This function generates three files:
        1. A text file containing the chapter's text.
        2. A JSON file containing the chapter's hierarchy.
        3. A JSON file containing the extracted paragraphs.

        Each file is named using the chapter index and the safe chapter title to ensure uniqueness and avoid file system issues.
    """

    os.makedirs(os.path.join(output_dir, 'text'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'summaries'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'hierarchy'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'paragraphs'), exist_ok=True)


    text_filename = f"{i}_{safe_chapter_title}.txt"
    text_filepath = os.path.join(output_dir, 'text', text_filename)
    safe_write_file(text, text_filepath, file_type='text')

    if summary_text:
        summary_filename = f"{i}_{safe_chapter_title}_summary.json"
        summary_filepath = os.path.join(output_dir, 'summaries', summary_filename)
        safe_write_file(summary_text, summary_filepath)

    hierarchy_filename = f'{i+1}_{safe_chapter_title}_hierarchy.json'
    hierarchy_filepath = os.path.join(output_dir, 'hierarchy', hierarchy_filename)
    safe_write_file(hierarchy, hierarchy_filepath)

    paragraphs_filename = f'{i+1}_{safe_chapter_title}_paragraphs.json'
    paragraphs_filepath = os.path.join(output_dir, 'paragraphs', paragraphs_filename)
    safe_write_file(paragraphs, paragraphs_filepath)

def process_book(
    book_path: str
) -> None:
    """
        Processes a single book by reading its EPUB file, extracting the table of contents, 
        chapter metadata, and paragraphs, and saving the extracted content to the specified 
        output directory.

        Parameters:
        - book_path (str): The path to the EPUB file to be processed.

        This function performs the following steps:
        1. Reads the EPUB file using the `epub` library.
        2. Maps the table of contents with create_toc_mapping.
            Creates a mapping of the table of contents (TOC) for the EPUB book. 
            This function extracts navigation points from the EPUB's NCX file and 
            returns a dictionary that maps content source paths to their corresponding 
            chapter titles. If no navigation item is found, it returns None.
        3. Eliminates fragments from the TOC mapping with eliminate_fragments.
            Combines the TOC mapping with a filtered version that eliminates fragments 
            (incomplete chapters indicated by a '#' in the file path). This results in 
            a clean mapping of complete chapters to their titles, ensuring that only 
            valid chapters are processed.
        4. Converts the TOC mapping to text version of the chapter with toc_to_text.
            Converts the TOC of the EPUB book into a formatted text representation. 
            This includes the book's title and author, along with a structured list 
            of chapters and sub-chapters, making it easier to understand the book's 
            organization.
        5. Extracts metadata from the book with extract_metadata.
            Extracts metadata from the EPUB book, including the title, creator, 
            language, identifier, publisher, and date. This information is crucial 
            for understanding the book's context and is used for naming output files 
            and organizing the extracted content.
        6. Sets up an output directory based on the book's title with setup_output_directory.
        6. Iterates through the items in the book, processing each chapter and 
        consolidating the extracted paragraphs.
        7. Saves the consolidated paragraphs to a JSON file. The 
        8. Returns the book name.

        Returns:
        - None: This function does not return any value but saves the output to files.
    """
    logging.info(f"Processing book: {book_path}")
    book = epub.read_epub(book_path)
    
    toc_mapping = create_toc_mapping(book)
    chapter_mapping = {**toc_mapping, **eliminate_fragments(toc_mapping)}
    decoded_chapter_mapping = decode_chapter_mapping(chapter_mapping)
    toc_text = toc_to_text(book)
    metadata = extract_metadata(book)

    logging.info(f"TOC text: {toc_text}")
    logging.info(f"Decoded chapter mapping: {json.dumps(decoded_chapter_mapping, indent=2)}")
    # logging.info(f"Metadata: {json.dumps(metadata, indent=2)}")

    all_paragraphs = []

    book_name = metadata['title'].replace(' ', '_')
    output_dir = setup_output_directory(book_name)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        process_chapter_partial = partial(
            process_chapter, 
            chapter_mapping=decoded_chapter_mapping, 
            metadata=metadata, 
            output_dir=output_dir
        )
        future_to_item = {
            executor.submit(process_chapter_partial, item=item, i=i): i 
            for i, item in enumerate(book.get_items()) 
            if item.get_type() == ebooklib.ITEM_DOCUMENT
        }
        
        for future in concurrent.futures.as_completed(future_to_item):
            paragraphs = future.result()
            all_paragraphs.extend(paragraphs)

    embedded_paragraphs = embed_dict_list(
        embedding_function=create_openai_embedding,
        chunk_dicts=all_paragraphs,
        model_choice = 'text-embedding-3-large',
        metadata_keys=[ 'chapter', 'title', 'author']
    )

    book_paragraphs_filepath = _save_consolidated_paragraphs(embedded_paragraphs, book_name, output_dir)
    logging.info(f"Finished processing book: {book_path}")

    result = {
        "name": book_name, 
        "paragraphs_filepath": book_paragraphs_filepath
    }

    return result

def _save_consolidated_paragraphs(
    all_paragraphs: List[Dict[str, Any]],
    book_name: str,
    output_dir: str
) -> None:
    """
        Saves the consolidated paragraphs to a JSON file.

        Parameters:
        - all_paragraphs (List[Dict[str, Any]]): A list of dictionaries containing the extracted paragraphs.
        - book_name (str): The name of the book, used to create the output filename.
        - output_dir (str): The directory where the output file will be saved.

        This function performs the following steps:
        1. Constructs the filename for the consolidated paragraphs using the book name.
        2. Joins the output directory and filename to create the full file path.
        3. Writes the list of paragraphs to a JSON file at the specified path.
        4. Logs the location of the saved file.

        Returns:
        - None: This function does not return any value but saves the output to a file.
    """
    logging.info(f"Number of paragraphs: {len(all_paragraphs)}")
    consolidated_paragraphs_filename = f'{book_name}_all_paragraphs.json'
    consolidated_paragraphs_filepath = os.path.join(output_dir, consolidated_paragraphs_filename)
    safe_write_file(all_paragraphs, consolidated_paragraphs_filepath)
    logging.info(f"Consolidated paragraphs saved to: {consolidated_paragraphs_filepath}")

    return consolidated_paragraphs_filepath

def process_books(
    book_paths: List[str]
) -> str:
    """
        Processes a list of book paths by extracting metadata, 
        processing each book, saving the consolidated paragraphs,
        and then combining all consolidated paragraphs into a single file.

        Args:
            book_paths (List[str]): List of paths to the EPUB files to be processed.

        Returns:
            str: The filepath of the combined paragraphs file, or None if no paragraphs were found.
    """
    book_paragraphs = []
    for book_path in book_paths:
        try:
            book_dict = process_book(book_path)
            name = book_dict['name']
            paragraphs_filepath = book_dict['paragraphs_filepath']
            if paragraphs_filepath:
                book_paragraphs.append({
                    "name": name,
                    "paragraphs_filepath": paragraphs_filepath
                })
                logging.info(f"Book {name} paragraphs complete")
        except Exception as e:
            logging.error(f"Error processing book {book_path}: {str(e)}")
    
    return book_paragraphs

def combine_consolidated_paragraphs(
    book_paragraphs_filepaths: List[str]
) -> str:
    """
        Combines consolidated paragraph files from multiple books into a single file.

        Args:
            book_paragraphs_filepaths (List[str]): List of filepaths to the consolidated paragraph files.

        Returns:
            str: The filepath of the combined paragraphs file.

        This function does the following:
        1. Iterates through the consolidated paragraph files.
        2. Reads the paragraphs from each file.
        3. Combines all paragraphs into a single list.
        4. Saves the combined list to a new JSON file.
    """
    all_paragraphs = []
    for filepath in book_paragraphs_filepaths:
        logging.info(f"Filepath: {filepath}")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                book_paragraphs = json.load(f)
                logging.info(f"Length of book paragraphs: {len(book_paragraphs)}")
                all_paragraphs.extend(book_paragraphs)
                logging.info(f"Length of all paragraphs: {len(all_paragraphs)}")
    
    if all_paragraphs:
        combined_file_path = os.path.join(EXTRACTED_DIR, "all_books_paragraphs.json")
        safe_write_file(all_paragraphs, combined_file_path)
        logging.info(f"Combined paragraphs from all books saved to: {combined_file_path}")
        return combined_file_path
    else:
        logging.warning("No paragraphs found to combine.")
        return None

def create_index(
    file_path: str, 
    index_dir: str = EXTRACTED_DIR
) -> None:
    """
        Creates an index of books and their chapters from a specified JSON file.

        Args:
            file_path (str): The path to the JSON file containing book and chapter data.
            index_dir (str, optional): The directory where the index file will be saved. 
                                        Defaults to EXTRACTED_DIR.

        This function performs the following steps:
        1. Calls the _filter_books_and_chapters function to extract and organize book titles 
           and their corresponding chapters from the provided JSON file.
        2. Constructs the path for the index file by joining the index_dir with the filename 
           "book_and_chapter_index.json".
        3. Saves the resulting dictionary from the filtering process to the index file using 
           the safe_write_file function.
        4. Logs the location where the index file has been saved.

        Returns:
            None: This function does not return a value, but it creates an index file 
            containing the organized book and chapter data.
    """
    book_and_chapter_dict = _filter_books_and_chapters(file_path)
    index_file_path = os.path.join(index_dir, "book_and_chapter_index.json")
    safe_write_file(book_and_chapter_dict, index_file_path)
    logging.info(f"Book and chapter index saved to: {index_file_path}")

def _filter_books_and_chapters(
    json_file_path: str
) -> Dict[str, Any]:
    """
        Filters the books and chapters from a JSON file.

        Args:
            json_file_path (str): The path to the JSON file containing book and chapter data.

        Returns:
            Dict[str, Any]: A dictionary containing a sorted list of unique book titles and 
            a dictionary of chapters for each book, sorted in natural order.

        This function performs the following steps:
        1. Opens the specified JSON file and loads its content.
        2. Initializes a set to store unique book titles and a defaultdict to store chapters.
        3. Iterates through the loaded data, extracting titles and chapters.
        4. Sorts the list of books and the chapters for each book using a natural sort key.
        5. Returns a dictionary containing the sorted list of books and the sorted chapters.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    books = set()
    chapters = defaultdict(set)

    for item in data:
        if 'title' in item and 'chapter' in item:
            books.add(item['title'])
            chapters[item['title']].add(item['chapter'])

    result = {
        "books": sorted(list(books)),
        "chapters": {book: sorted(list(chapter_set), key=_natural_sort_key) 
                     for book, chapter_set in sorted(chapters.items())}
    }

    return result

def _natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def update_book_paragraphs_filepaths(
    book_paths: List[str],
    paragraphs_filepath: str = 'book_paragraphs_filepaths.json'
) -> None:
    """
        Updates the JSON file containing the file paths of book paragraphs.

        This function takes a list of book paths and a file path to a JSON file that stores
        the mapping of book names to their corresponding paragraph file paths. It retrieves
        the current mappings, processes the provided book paths to extract their names and
        corresponding paragraph file paths, and updates the JSON file with the new mappings.

        Parameters:
        - book_paths (List[str]): A list of file paths to the books for which paragraph
          file paths need to be updated.
        - paragraphs_filepath (str): The path to the JSON file that contains the current
          mappings of book names to their paragraph file paths. Defaults to 'book_paragraphs_filepaths.json'.

        Returns:
        - None: This function does not return any value. It modifies the JSON file in place.

        Steps:
        1. Retrieve the current book paragraphs file paths from the specified JSON file.
        2. Create a deep copy of the current mappings to avoid modifying the original data directly.
        3. Process the provided book paths to extract the book names and their corresponding
           paragraph file paths.
        4. Update the deep copied dictionary with the new mappings.
        5. Write the updated dictionary back to the JSON file, effectively updating the mappings
           of book names to their paragraph file paths.
    """
    book_paragraphs_filepaths = retrieve_file(paragraphs_filepath)

    new_dict = copy.deepcopy(book_paragraphs_filepaths)
    
    book_paragraphs = process_books(book_paths)
    for book in book_paragraphs:
        new_dict[book['name']] = book['paragraphs_filepath']

    write_to_file(new_dict, paragraphs_filepath)

def load_and_combine_paragraphs(
    books_to_load: List[str],
    paragraphs_filepath: str
) -> None:
    """
        Loads and combines paragraphs from specified books into a single consolidated file.

        This function takes a list of book names to load and a file path to a JSON file that contains
        the mappings of book names to their corresponding paragraph file paths. It retrieves the file paths
        for the specified books, combines the contents of these files, and creates an index for the combined
        paragraphs.

        Parameters:
        - books_to_load (List[str]): A list of book names that need to be loaded and combined.
        - paragraphs_filepath (str): The path to the JSON file that contains the mappings of book names
          to their corresponding paragraph file paths.

        Returns:
        - None: This function does not return any value. It modifies the output by creating a consolidated
          file of paragraphs and an index for it.

        Steps:
        1. Retrieve the current book paragraphs file paths from the specified JSON file using the
           `retrieve_file` function.
        2. Initialize an empty list to store the file paths of the books to be loaded.
        3. Iterate over the list of books to load:
            - For each book, append its corresponding file path from the retrieved mappings to the list
              of file paths to load.
        4. Call the `combine_consolidated_paragraphs` function with the list of file paths to load,
           which combines the contents of these files into a single consolidated file.
        5. Create an index for the combined paragraphs by calling the `create_index` function, passing
           the path of the combined file and the directory where the index should be stored.
    """
    book_paragraphs_filepaths = retrieve_file(paragraphs_filepath)

    filepaths_to_load = []
    
    for book in books_to_load:
        filepaths_to_load.append(book_paragraphs_filepaths[book])
    
    combined_file_path = combine_consolidated_paragraphs(filepaths_to_load)
    
    create_index(combined_file_path, index_dir=EXTRACTED_DIR)


if __name__ == "__main__":

    if False:
        book_paths = [
            '../The Inevitable -- Kevin Kelly -- 2016 -- Penguin Publishing Group copy.epub'
        ]

        update_book_paragraphs_filepaths(book_paths, 'book_paragraphs_filepaths.json')

    else:
        books_to_load = [
            "The_Inevitable"
        ]

        load_and_combine_paragraphs(books_to_load, 'book_paragraphs_filepaths.json')


     