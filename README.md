# Hr_interview_preperationAgent

## Introduction
This project extracts text from PDF textbooks, generates hierarchical embeddings using Sentence-BERT, and stores them in a PostgreSQL database for retrieval and analysis.

## Prerequisites
Ensure you have the following installed:
- Python 3.8 or later
- PostgreSQL with the `vector` extension
- Required Python packages (see below)

## Installation & Setup

### 1. Clone the Repository
```sh
git clone https://github.com/zarouz/Hr_interview_preperationAgent/tree/knowledgeBase
```

### 2. Install Required Packages
```sh
pip install -r requirements.txt
```

**Dependencies include:**
- `psycopg2`
- `sentence-transformers`
- `nltk`
- `pdfplumber`
- `numpy`
- `python-dotenv`

### 3. Set Up PostgreSQL Database
Ensure PostgreSQL is running and create the necessary schema:
```sh
psql -U <your-username> -d KnowledgeBase -f schema.sql
```
Alternatively, run `setup_database()` in the script to create tables automatically.

### 4. Run the Script
To parse a textbook directory and store embeddings, execute:
```sh
python main.py /path/to/textbook/folder
```

## Textbook Storage Guidelines
To ensure proper processing, store textbooks in the following format:
- Place all textbooks inside the `textbooks/` folder.
- Each textbook should have its own subfolder named after the book title.
- Chapters should be stored as separate PDF files with a consistent naming convention:
  ```
  textbooks/
    ├── Book_Title/
    │   ├── Introduction_1.pdf
    │   ├── Basics_2.pdf
    │   ├── Advanced_Topics_3.pdf
    │   └── ...
  ```
- Use a consistent naming format (`name_number.pdf`).
- Avoid spaces in filenames; use underscores (`_`) instead.

## Bookmaker: Structuring the Textbook
The `bookmaker.py` utility helps in structuring textbooks by splitting PDFs into structured chapters based on user input. It uses PyPDF2 to divide PDFs into meaningful sections.

## Database Availability & Usage
The database containing embeddings of the OS book is available for download at the following Google Drive link:
[Download Database](https://drive.google.com/file/d/1aRXbG5xhxtRoP6ipGU_NjEQqNc7dpdR-/view?usp=drive_link)

### Restoring the Database
To restore the database from the provided SQL dump file, use the following command:
```sh
psql -U <your-username> -d <your-database-name> -f knowledgeBase.sql
```

Example:
```sh
psql -U karthikyadav -d KnowledgeBase -f knowledgeBase.sql
```

### Uploading the Database Backup
If you wish to upload the database after making changes, export it using:
```sh
pg_dump -U <your-username> -d <your-database-name> -f knowledgeBase.sql
```

Example:
```sh
pg_dump -U karthikyadav -d KnowledgeBase -f knowledgeBase.sql
```

This will create a backup file that can be shared or uploaded for others to use.

## Usage Guide
- **Parsing PDFs**: The script expects textbooks in a structured folder.
- **Storing Embeddings**: The Sentence-BERT model generates embeddings for hierarchical text units.
- **Retrieving Data**: Query the database using vector similarity search on `chunks.embedding`.

## Troubleshooting
- If `vector` extension errors occur, install it in PostgreSQL:
  ```sh
  psql -U <your-username> -d KnowledgeBase -c "CREATE EXTENSION IF NOT EXISTS vector;"
  ```
- Ensure your `.env` file is correctly set up.
- Verify `nltk_data` path matches your system setup.

## Contact
For issues or improvements, submit a pull request or open an issue in the repository.
