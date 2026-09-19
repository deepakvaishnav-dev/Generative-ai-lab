import sys
import os
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

sys.stdout.reconfigure(encoding='utf-8')

# Path pointing to the document-loader folder
pdf_path = os.path.join(os.path.dirname(__file__), "..", "document-loader", "Python_Complete_Notes.pdf")

data = PyPDFLoader(pdf_path)
docs = data.load()

splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

# Split the loaded document into token chunks
chunks = splitter.split_documents(docs)

print(f"Total pages loaded: {len(docs)}")
print(f"Total token chunks created: {len(chunks)}")
print("\n--- First Chunk Preview ---")
print(chunks[0].page_content)
