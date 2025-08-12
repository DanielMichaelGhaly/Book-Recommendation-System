# Book-Recommendation-System

# Installation

## Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

## Required Packages

Install the required packages using pip:

```bash
pip install pandas numpy python-dotenv langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-core chromadb gradio
```


## Additional Requirements

Make sure you have the following files in your project directory:
- `books_with_emotions.csv` - CSV file containing book data with emotions
- `tagged_description.txt` - Text file with tagged book descriptions
- `cover-not-found.jpg` - Placeholder image for books without covers
- `.env` - Environment variables file (if using any API keys)

## Running the Application

After installing the dependencies, you can run the application with:

```bash
python gradio-dashboard.py
```

The application will be available at `http://127.0.0.1:7860` in your web browser.
