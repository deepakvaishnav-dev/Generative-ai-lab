import sys
import os
from langchain_community.document_loaders import PyPDFLoader

# Fix Unicode encoding for Windows terminal (handles emojis and special chars)
sys.stdout.reconfigure(encoding='utf-8')

# Build absolute path relative to this script's location
pdf_path = os.path.join(os.path.dirname(__file__), "Python_Complete_Notes.pdf")

data = PyPDFLoader(pdf_path)

docs = data.load()

print(docs)