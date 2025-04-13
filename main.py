import os
import re
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import pdfplumber  # For PDF text extraction
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from .env (if available)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set NLTK data directory so it finds our downloaded resources
# --- IMPORTANT: Update this path if necessary ---
nltk_data_path = "/Users/karthikyadav/nltk_data"
if os.path.exists(nltk_data_path):
    nltk.data.path.append(nltk_data_path)
else:
    logger.warning(f"NLTK data path not found: {nltk_data_path}. Downloads might go to default location.")

# Download necessary NLTK resources (quietly)
try:
    nltk.download('punkt', quiet=True)
    # nltk.download('punkt_tab', quiet=True) # punkt_tab seems less common, often covered by punkt
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")


# --- Step 1: Database Connection and Schema Setup ---
def setup_database():
    """Connect to PostgreSQL and create the necessary tables and extensions."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "KnowledgeBase"),
            user=os.getenv("DB_USER", "karthikyadav"),
            password=os.getenv("DB_PASSWORD", "") # Use env var or set password directly if needed
        )
        cursor = conn.cursor()

        # Create vector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create textbooks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS textbooks (
                id SERIAL PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                author VARCHAR(255),
                description TEXT,
                folder_path VARCHAR(512) UNIQUE, -- Added UNIQUE constraint
                publication_date DATE,
                embedding VECTOR(768) -- Assuming all-mpnet-base-v2 dimension
            );
        """)

        # Create chapters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chapters (
                id SERIAL PRIMARY KEY,
                textbook_id INT REFERENCES textbooks(id) ON DELETE CASCADE, -- Added ON DELETE CASCADE
                chapter_number INT,
                title VARCHAR(255),
                file_path VARCHAR(512) UNIQUE, -- Added UNIQUE constraint
                embedding VECTOR(768) -- Assuming all-mpnet-base-v2 dimension
            );
        """)

        # Create hierarchical chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                textbook_id INT REFERENCES textbooks(id) ON DELETE CASCADE, -- Added ON DELETE CASCADE
                chapter_id INT REFERENCES chapters(id) ON DELETE CASCADE, -- Added ON DELETE CASCADE
                parent_chunk_id INT REFERENCES chunks(id) ON DELETE CASCADE, -- Added ON DELETE CASCADE
                chunk_type VARCHAR(50) NOT NULL CHECK (chunk_type IN ('page', 'paragraph', 'sentence')), -- Added CHECK constraint
                content TEXT NOT NULL,
                embedding VECTOR(768) NOT NULL, -- Assuming all-mpnet-base-v2 dimension
                page_number INT,
                start_offset INT, -- Keep if needed for future use
                end_offset INT    -- Keep if needed for future use
            );
        """)

        # Create indices for faster similarity search
        # Note: Index creation might take time on large tables
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_textbooks_embedding ON textbooks
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chapters_embedding ON chapters
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
        """)

        # Index for faster lookups by path (helps prevent duplicates)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_textbooks_folder_path ON textbooks (folder_path);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chapters_file_path ON chapters (file_path);")


        conn.commit()
        logger.info("Database connection successful and schema verified/created.")
        return conn, cursor

    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("Please ensure PostgreSQL is running and the connection details (host, db, user, password) are correct.")
        raise # Re-raise the exception to stop execution if DB connection fails
    except Exception as e:
        logger.error(f"An error occurred during database setup: {e}")
        raise

# --- Step 2: Parse Textbook Structure ---
def parse_textbook_structure(textbook_dir):
    """
    Parse the textbook directory structure to identify chapters.
    Expected structure:
    - textbook_folder/
      - index.pdf (optional)
      - chapter_name_1.pdf
      - chapter_name_with_underscores_2.pdf
      - ...
    """
    if not os.path.isdir(textbook_dir):
        logger.error(f"Directory not found: {textbook_dir}")
        raise ValueError(f"Directory not found: {textbook_dir}")

    # Extract textbook title from folder name (can be improved if needed)
    textbook_title = os.path.basename(textbook_dir)
    logger.info(f"Parsing textbook structure for: {textbook_title}")

    # Find all PDF files
    try:
        pdf_files = [f for f in os.listdir(textbook_dir) if f.lower().endswith('.pdf') and not f.startswith('.')]
    except OSError as e:
        logger.error(f"Error listing files in directory {textbook_dir}: {e}")
        raise

    index_file = None
    chapter_files = []

    for pdf in pdf_files:
        pdf_lower = pdf.lower()
        if pdf_lower == 'index.pdf':
            index_file = os.path.join(textbook_dir, pdf)
            logger.info(f"Found index file: {pdf}")
        else:
            # Parse chapter number from filename (format: chaptername_chapternumber.pdf)
            # Regex explanation:
            # _           - Matches the literal underscore
            # (\d+)       - Matches one or more digits (captures the chapter number)
            # \.pdf$      - Matches the literal ".pdf" at the end of the string
            # re.IGNORECASE - Makes the matching case-insensitive for ".pdf"
            match = re.search(r'_(\d+)\.pdf$', pdf, re.IGNORECASE)
            if match:
                chapter_number = int(match.group(1))

                # --- CORRECTED CHAPTER NAME EXTRACTION ---
                # Use rsplit to split from the right at the last underscore
                parts = pdf.rsplit('_', 1)
                if len(parts) == 2:
                    # Take the part before the last underscore
                    chapter_name = parts[0].replace('_', ' ').strip() # Replace underscores with spaces for readability
                else:
                    # Fallback: if no underscore found just before number (unlikely with the regex)
                    # Try removing the matched suffix (_number.pdf)
                    chapter_name = pdf.replace(match.group(0), "").strip()
                    logger.warning(f"Could not reliably split chapter name for {pdf}. Using fallback: '{chapter_name}'")
                # --- END CORRECTION ---

                chapter_files.append({
                    'file_path': os.path.join(textbook_dir, pdf),
                    'chapter_number': chapter_number,
                    'title': chapter_name  # Use the correctly extracted title
                })
                # logger.debug(f"Found chapter: '{chapter_name}' (Number: {chapter_number}) Path: {pdf}") # More verbose logging if needed
            else:
                logger.warning(f"Could not parse chapter number from filename: {pdf}. Skipping this file.")

    # Sort chapters by chapter number
    chapter_files.sort(key=lambda x: x['chapter_number'])

    textbook_info = {
        'title': textbook_title,
        'folder_path': textbook_dir,
        'index_file': index_file,
        'chapters': chapter_files,
        'author': "Unknown", # Default values
        'description': "",
        'publication_date': None
    }

    # Extract metadata from index.pdf if available
    if index_file:
        logger.info(f"Extracting metadata from index file: {index_file}")
        try:
            author, description, publication_date = extract_metadata_from_index(index_file)
            textbook_info['author'] = author if author else "Unknown"
            textbook_info['description'] = description if description else ""
            textbook_info['publication_date'] = publication_date
            logger.info(f"Metadata extracted: Author='{textbook_info['author']}', Date='{textbook_info['publication_date']}'")
        except Exception as e:
            logger.error(f"Failed to extract metadata from {index_file}: {e}")
            # Keep defaults if extraction fails

    return textbook_info

def extract_metadata_from_index(index_file):
    """Extract author, description, and publication date from index.pdf (basic implementation)."""
    author = None
    description = ""
    publication_date = None # Use None as default, store only if found

    try:
        with pdfplumber.open(index_file) as pdf:
            # Extract text from the first few pages (e.g., first 3) for efficiency
            text = ""
            num_pages_to_scan = min(3, len(pdf.pages))
            for i in range(num_pages_to_scan):
                page = pdf.pages[i]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            # Simple pattern matching for metadata - enhance as needed
            # Look for lines starting with "Author(s):" etc.
            author_match = re.search(r'^(?:Author|Authors)\s*[:\-]\s*(.+)$', text, re.IGNORECASE | re.MULTILINE)
            if author_match:
                author = author_match.group(1).strip()

            # Extract a description (e.g., first reasonably long paragraph, or text after "Abstract"/"Summary")
            # This is highly dependent on PDF structure
            # Let's take the first few non-empty lines as a basic description
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            description = " ".join(lines[:5])[:500] # Limit length

            # Try to find a date (YYYY-MM-DD, DD/MM/YYYY, Month YYYY, or just YYYY)
            # More robust date parsing might be needed
            date_patterns = [
                r'(\d{4}-\d{1,2}-\d{1,2})',             # YYYY-MM-DD
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',       # DD/MM/YYYY or DD-MM-YYYY
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}', # Month YYYY
                r'\b(19\d{2}|20\d{2})\b'                 # YYYY (standalone year)
            ]
            found_date_str = None
            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    found_date_str = date_match.group(0) # Use the first match found
                    break

            if found_date_str:
                try:
                    # Attempt parsing common formats
                    if re.match(r'\d{4}-\d{1,2}-\d{1,2}', found_date_str):
                       publication_date = datetime.strptime(found_date_str, '%Y-%m-%d').date()
                    elif re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', found_date_str):
                        publication_date = datetime.strptime(found_date_str.replace('-', '/'), '%d/%m/%Y').date()
                    elif re.match(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}', found_date_str, re.IGNORECASE):
                         # Simplified: Use the year part if month format is complex
                         year_match = re.search(r'\d{4}', found_date_str)
                         if year_match:
                             publication_date = datetime(int(year_match.group(0)), 1, 1).date() # Default to Jan 1st
                    elif re.match(r'\b(19\d{2}|20\d{2})\b', found_date_str):
                        publication_date = datetime(int(found_date_str), 1, 1).date() # Default to Jan 1st

                except ValueError as date_err:
                    logger.warning(f"Could not parse extracted date string '{found_date_str}': {date_err}")
                    publication_date = None # Reset if parsing fails

    except Exception as e:
        logger.error(f"Error extracting metadata from index file {index_file}: {e}")

    # Return None if not found, otherwise the date object
    return author, description, publication_date


# --- Step 3: Extract text from PDF chapters ---
def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file, returning a list of strings (one per page)."""
    text_by_page = []
    logger.info(f"Extracting text from: {os.path.basename(pdf_path)}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                logger.warning(f"PDF file has no pages: {pdf_path}")
                return [""] # Return list with empty string if no pages

            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    # Basic cleaning: replace multiple whitespace chars with a single space
                    cleaned_text = re.sub(r'\s+', ' ', page_text).strip() if page_text else ""
                    text_by_page.append(cleaned_text)
                    # logger.debug(f"Extracted text from page {i+1} of {os.path.basename(pdf_path)}")
                except Exception as page_err:
                    logger.error(f"Error extracting text from page {i+1} in {pdf_path}: {page_err}")
                    text_by_page.append("") # Append empty string for problematic pages
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        return ["Error: PDF file not found"]
    except Exception as e:
        logger.error(f"Error opening or processing PDF file {pdf_path}: {e}")
        # Return an indication of error for all potential pages
        # Adjust this based on how you want to handle partial failures
        return [f"Error extracting text from PDF: {e}"]

    logger.info(f"Finished extracting text from {os.path.basename(pdf_path)}. Pages found: {len(text_by_page)}")
    return text_by_page


def extract_paragraphs(text):
    """Split text into paragraphs, attempting to handle common PDF extraction issues."""
    if not text:
        return []

    # 1. Split by double newlines (common paragraph marker)
    paragraphs = re.split(r'\n\s*\n', text)

    # 2. Further processing to handle potential single newlines within paragraphs
    #    and clean up whitespace.
    cleaned_paragraphs = []
    for p in paragraphs:
        # Replace single newlines (often due to line breaks in PDF) with spaces
        p_cleaned = p.replace('\n', ' ')
        # Consolidate multiple spaces and strip leading/trailing whitespace
        p_cleaned = re.sub(r'\s+', ' ', p_cleaned).strip()
        if len(p_cleaned) > 15: # Only keep paragraphs with some substance
            cleaned_paragraphs.append(p_cleaned)

    # 3. Alternative/Fallback: If double newline splitting yields few results,
    #    try splitting by lines that seem to end sentences (e.g., end with .?!).
    #    This is less reliable.
    if not cleaned_paragraphs or len(cleaned_paragraphs) < 3: # Heuristic threshold
        lines = text.splitlines()
        potential_paragraphs = []
        current_para = ""
        for line in lines:
            line = line.strip()
            if line:
                current_para += line + " "
                if line.endswith(('.', '!', '?')): # Simple sentence end detection
                    if len(current_para.strip()) > 15:
                         potential_paragraphs.append(current_para.strip())
                    current_para = ""
        if current_para.strip() and len(current_para.strip()) > 15: # Add any remaining part
            potential_paragraphs.append(current_para.strip())

        # Use the potential paragraphs only if they seem better than the original split
        if len(potential_paragraphs) > len(cleaned_paragraphs):
             # logger.debug("Using sentence-ending based paragraph splitting as fallback.")
             return potential_paragraphs

    return cleaned_paragraphs


# --- Step 4: Generate embeddings ---
def initialize_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    """Initialize the sentence transformer model for embeddings."""
    logger.info(f"Initializing embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load sentence transformer model '{model_name}': {e}")
        raise # Stop execution if model can't load


def create_hierarchical_embeddings(chapter_text_by_page, model):
    """
    Create hierarchical embeddings for chapter text: Page, Paragraph, Sentence.
    Returns a structured dictionary.
    """
    if not chapter_text_by_page or all(not page_text for page_text in chapter_text_by_page):
        logger.warning("No text content found for this chapter. Skipping embedding generation.")
        return {'pages': []} # Return empty structure

    hierarchy = {'pages': []}
    total_sentences = 0
    total_paragraphs = 0

    # Process each page
    for page_idx, page_text in enumerate(chapter_text_by_page):
        page_num = page_idx + 1 # 1-based page numbering
        if not page_text or not page_text.strip():
            # logger.debug(f"Skipping empty page {page_num}")
            continue

        # logger.debug(f"Processing Page {page_num}")

        # Create page-level embedding
        # Consider adding context like "Page X of Chapter Y" if desired
        try:
            page_embedding = model.encode(page_text)
        except Exception as e:
            logger.error(f"Error encoding page {page_num}: {e}")
            continue # Skip page if encoding fails

        page_data = {
            'page_number': page_num,
            'embedding': page_embedding,
            'content': page_text, # Store original cleaned page text
            'paragraphs': []
        }

        # Extract paragraphs from the page
        paragraphs = extract_paragraphs(page_text)
        if not paragraphs:
            # logger.debug(f"No paragraphs extracted from page {page_num}")
            # Still add the page chunk itself if it has content
            if page_data['content']:
                 hierarchy['pages'].append(page_data)
            continue

        # Process paragraphs
        for para_idx, para_text in enumerate(paragraphs):
            # Basic check for paragraph validity (adjust length as needed)
            if len(para_text) < 20:
                # logger.debug(f"Skipping short paragraph on page {page_num}: '{para_text[:30]}...'")
                continue

            # Create paragraph-level embedding
            try:
                para_embedding = model.encode(para_text)
            except Exception as e:
                logger.error(f"Error encoding paragraph {para_idx+1} on page {page_num}: {e}")
                continue # Skip paragraph if encoding fails

            # Extract sentences
            try:
                sentences = sent_tokenize(para_text)
            except Exception as e:
                logger.error(f"NLTK sent_tokenize failed for paragraph on page {page_num}: {e}")
                sentences = [] # Handle tokenization failure gracefully

            para_data = {
                'embedding': para_embedding,
                'content': para_text,
                'sentences': []
            }

            # Process sentences
            for sent_idx, sent_text in enumerate(sentences):
                sent_text = sent_text.strip()
                # Basic check for sentence validity (adjust length as needed)
                if len(sent_text) < 10:
                    # logger.debug(f"Skipping short sentence: '{sent_text}'")
                    continue

                # Create sentence-level embedding
                try:
                    sent_embedding = model.encode(sent_text)
                except Exception as e:
                    logger.error(f"Error encoding sentence {sent_idx+1} in paragraph {para_idx+1} on page {page_num}: {e}")
                    continue # Skip sentence if encoding fails

                # Add sentence data
                para_data['sentences'].append({
                    'embedding': sent_embedding,
                    'content': sent_text
                })

            # Only add paragraph if it has valid content and maybe sentences
            if para_data['content']: # Or check: if para_data['sentences']:
                page_data['paragraphs'].append(para_data)
                total_paragraphs += 1
                total_sentences += len(para_data['sentences'])


        # Only add page if it has meaningful content (e.g., valid paragraphs)
        if page_data['paragraphs'] or (not paragraphs and page_data['content']): # Add page if it has paragraphs OR if it had content but no paras were extracted
            hierarchy['pages'].append(page_data)

    logger.info(f"Generated hierarchical embeddings: {len(hierarchy['pages'])} pages, {total_paragraphs} paragraphs, {total_sentences} sentences.")
    return hierarchy


# --- Step 5: Store in database ---
def store_textbook_and_hierarchical_embeddings(conn, cursor, textbook_info, model):
    """Store the textbook, chapters, and hierarchical embeddings in the database, checking for duplicates."""

    # --- Check if textbook already exists ---
    cursor.execute("SELECT id FROM textbooks WHERE folder_path = %s", (textbook_info['folder_path'],))
    existing_textbook = cursor.fetchone()
    if existing_textbook:
        logger.warning(f"Textbook '{textbook_info['title']}' with path '{textbook_info['folder_path']}' already exists in the database (ID: {existing_textbook[0]}). Skipping insertion.")
        # Optionally: Update existing textbook metadata here if needed
        # Or: Delete existing and re-insert (use with caution!)
        # For now, we just skip.
        return # Exit the function if textbook exists

    # --- Store Textbook ---
    # Create a textbook summary for embedding
    textbook_summary = (
        f"Textbook Title: {textbook_info['title']}. "
        f"Author(s): {textbook_info.get('author', 'Unknown')}. "
        f"Description: {textbook_info.get('description', 'No description available.')}"
    )
    logger.info(f"Generating embedding for textbook: {textbook_info['title']}")
    try:
        textbook_embedding = model.encode(textbook_summary)
    except Exception as e:
        logger.error(f"Failed to generate embedding for textbook '{textbook_info['title']}': {e}")
        textbook_embedding = np.zeros(model.get_sentence_embedding_dimension()) # Store zero vector on failure? Or handle differently.


    try:
        cursor.execute("""
            INSERT INTO textbooks (title, author, description, folder_path, publication_date, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            textbook_info['title'],
            textbook_info.get('author', 'Unknown'),
            textbook_info.get('description', ''),
            textbook_info['folder_path'],
            textbook_info.get('publication_date'), # Pass None if not found
            textbook_embedding.tolist()
        ))
        textbook_id = cursor.fetchone()[0]
        logger.info(f"Stored textbook: '{textbook_info['title']}' with ID {textbook_id}")
        conn.commit() # Commit after successful textbook insertion
    except Exception as e:
        logger.error(f"Database error storing textbook '{textbook_info['title']}': {e}")
        conn.rollback() # Rollback if textbook insertion fails
        return # Stop processing this textbook

    # --- Process and Store Each Chapter ---
    for chapter in textbook_info['chapters']:
        logger.info(f"Processing Chapter {chapter['chapter_number']}: '{chapter['title']}'")

        # --- Check if chapter already exists ---
        cursor.execute("SELECT id FROM chapters WHERE file_path = %s", (chapter['file_path'],))
        existing_chapter = cursor.fetchone()
        if existing_chapter:
            logger.warning(f"Chapter '{chapter['title']}' with path '{chapter['file_path']}' already exists (ID: {existing_chapter[0]}). Skipping.")
            continue # Skip to the next chapter

        # Extract text from PDF
        chapter_text_by_page = extract_text_from_pdf(chapter['file_path'])
        if not chapter_text_by_page or "Error:" in chapter_text_by_page[0]:
             logger.error(f"Skipping chapter '{chapter['title']}' due to text extraction errors.")
             continue

        # Generate chapter-level embedding
        # Use a summary + start of content for chapter embedding
        chapter_summary = f"Chapter {chapter['chapter_number']}: {chapter['title']} from textbook {textbook_info['title']}."
        full_chapter_text = "\n\n".join(page for page in chapter_text_by_page if page and page.strip())
        chapter_embedding = np.zeros(model.get_sentence_embedding_dimension()) # Default to zero vector

        if full_chapter_text:
            # Limit context size for chapter embedding generation
            content_snippet = full_chapter_text[:4000] # Use first ~4000 chars
            embedding_text = chapter_summary + "\n\n" + content_snippet
            logger.info(f"Generating embedding for Chapter {chapter['chapter_number']}")
            try:
               chapter_embedding = model.encode(embedding_text)
            except Exception as e:
               logger.error(f"Failed to generate embedding for chapter '{chapter['title']}': {e}")
               # Keeps the zero vector

        # Store chapter metadata
        try:
            cursor.execute("""
                INSERT INTO chapters (textbook_id, chapter_number, title, file_path, embedding)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                textbook_id,
                chapter['chapter_number'],
                chapter['title'],
                chapter['file_path'],
                chapter_embedding.tolist()
            ))
            chapter_id = cursor.fetchone()[0]
            logger.info(f"Stored chapter: '{chapter['title']}' with ID {chapter_id}")
        except Exception as e:
            logger.error(f"Database error storing chapter '{chapter['title']}': {e}")
            conn.rollback() # Rollback this chapter's transaction
            continue # Skip to the next chapter

        # Generate hierarchical embeddings for the chapter content
        logger.info(f"Generating hierarchical embeddings for Chapter {chapter['chapter_number']}")
        hierarchy = create_hierarchical_embeddings(chapter_text_by_page, model)

        # Store hierarchical chunks (pages, paragraphs, sentences)
        page_count = 0
        para_count = 0
        sent_count = 0
        try:
            # Store page-level chunks
            for page_data in hierarchy.get('pages', []):
                page_content = page_data['content']
                page_embedding = page_data['embedding']
                page_number = page_data['page_number']

                cursor.execute("""
                    INSERT INTO chunks (textbook_id, chapter_id, parent_chunk_id, chunk_type,
                                       content, embedding, page_number)
                    VALUES (%s, %s, NULL, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    textbook_id, chapter_id, 'page',
                    page_content, page_embedding.tolist(), page_number
                ))
                page_chunk_id = cursor.fetchone()[0]
                page_count += 1

                # Store paragraph-level chunks
                for para_data in page_data.get('paragraphs', []):
                    para_content = para_data['content']
                    para_embedding = para_data['embedding']

                    cursor.execute("""
                        INSERT INTO chunks (textbook_id, chapter_id, parent_chunk_id, chunk_type,
                                           content, embedding, page_number)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (
                        textbook_id, chapter_id, page_chunk_id, 'paragraph',
                        para_content, para_embedding.tolist(), page_number
                    ))
                    para_chunk_id = cursor.fetchone()[0]
                    para_count += 1

                    # Store sentence-level chunks
                    for sent_data in para_data.get('sentences', []):
                        sent_content = sent_data['content']
                        sent_embedding = sent_data['embedding']

                        cursor.execute("""
                            INSERT INTO chunks (textbook_id, chapter_id, parent_chunk_id, chunk_type,
                                               content, embedding, page_number)
                            VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """, (
                            textbook_id, chapter_id, para_chunk_id, 'sentence',
                            sent_content, sent_embedding.tolist(), page_number
                        ))
                        sent_count += 1

            conn.commit() # Commit after successfully processing all chunks for the chapter
            logger.info(f"Stored chunks for chapter '{chapter['title']}': {page_count} pages, {para_count} paragraphs, {sent_count} sentences.")

        except Exception as e:
            logger.error(f"Database error storing chunks for chapter '{chapter['title']}': {e}")
            conn.rollback() # Rollback chunk insertions for this chapter on error
            # Consider deleting the chapter entry itself if chunks fail? Or leave it.
            logger.error(f"Rolling back chunk insertions for chapter ID {chapter_id}")
            # Optional: Delete the chapter record if its chunks failed?
            # cursor.execute("DELETE FROM chapters WHERE id = %s", (chapter_id,))
            # conn.commit()
            # logger.warning(f"Deleted chapter record ID {chapter_id} due to chunk insertion failure.")


    logger.info(f"Successfully finished processing for textbook '{textbook_info['title']}'")


# --- Main Processing Function ---
def process_textbook(textbook_dir, model, conn, cursor):
    """Processes a single textbook directory."""
    try:
        # Parse textbook structure
        textbook_info = parse_textbook_structure(textbook_dir)
        if not textbook_info['chapters']:
             logger.warning(f"No valid chapter files found in {textbook_dir}. Skipping.")
             return False

        logger.info(f"Processing textbook: '{textbook_info['title']}' with {len(textbook_info['chapters'])} chapters found.")

        # Store textbook and hierarchical embeddings
        store_textbook_and_hierarchical_embeddings(conn, cursor, textbook_info, model)

        return True

    except ValueError as ve: # Catch specific errors like directory not found
        logger.error(f"Skipping directory {textbook_dir}: {ve}")
        return False
    except Exception as e:
        logger.exception(f"An unexpected error occurred processing textbook at {textbook_dir}: {e}") # Log full traceback
        conn.rollback() # Rollback any partial transaction for this book
        return False


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    # Base directory containing all textbook subdirectories
    all_books_base_directory = "/Users/karthikyadav/Desktop/realtimeProjects/ml project/BuildKnowledgeBase/Books"

    # Option 1: Process ALL textbooks in the base directory
    process_all = True # Set to False to process only one specific textbook

    # Option 2: Process only ONE specific textbook (if process_all is False)
    specific_textbook_directory = "/Users/karthikyadav/Desktop/realtimeProjects/ml project/BuildKnowledgeBase/Books/ComputerNetworks" # Example

    # --- Initialization ---
    conn = None # Initialize connection variable
    try:
        # Setup database connection (do this once)
        conn, cursor = setup_database()

        # Initialize embedding model (do this once)
        model = initialize_embedding_model() # Use default model

        # --- Processing ---
        if process_all:
            logger.info(f"Starting processing for all textbooks in: {all_books_base_directory}")
            if not os.path.isdir(all_books_base_directory):
                 logger.error(f"Base directory not found: {all_books_base_directory}. Exiting.")
                 exit()

            items_in_base_dir = os.listdir(all_books_base_directory)
            processed_count = 0
            skipped_count = 0

            for item_name in items_in_base_dir:
                item_path = os.path.join(all_books_base_directory, item_name)
                # Process only if it's a directory and not hidden
                if os.path.isdir(item_path) and not item_name.startswith('.'):
                    logger.info(f"--- Processing textbook directory: {item_path} ---")
                    if process_textbook(item_path, model, conn, cursor):
                        processed_count += 1
                    else:
                        skipped_count += 1
                    logger.info(f"--- Finished processing: {item_path} ---")
                else:
                    logger.info(f"Skipping non-textbook directory/file: {item_name}")

            logger.info(f"Processing complete. Textbooks processed: {processed_count}, Skipped/Errored: {skipped_count}")

        else:
            # Process only the specific textbook directory
            logger.info(f"Starting processing for specific textbook: {specific_textbook_directory}")
            if os.path.isdir(specific_textbook_directory):
                 process_textbook(specific_textbook_directory, model, conn, cursor)
                 logger.info(f"Finished processing specific textbook.")
            else:
                 logger.error(f"Specific textbook directory not found: {specific_textbook_directory}")


    except (psycopg2.Error, ImportError, ValueError, RuntimeError, OSError) as E: # Catch specific expected errors
         logger.error(f"A critical error occurred during initialization or processing: {E}")
    except Exception as e:
        logger.exception(f"An unexpected critical error occurred: {e}") # Log full traceback for unexpected errors
    finally:
        # --- Cleanup ---
        if conn:
            try:
                conn.commit() # Final commit attempt in case something was pending
                logger.info("Final commit successful.")
            except psycopg2.Error as commit_err:
                 logger.error(f"Error during final commit: {commit_err}")
            if cursor:
                cursor.close()
            conn.close()
            logger.info("Database connection closed.")