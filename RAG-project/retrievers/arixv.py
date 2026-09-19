import sys
from langchain_community.retrievers import ArxivRetriever

# Ensure UTF-8 output in Windows terminal to avoid UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

# Create the retriever

retriever = ArxivRetriever(
    load_max_docs= 2,
    load_all_available_meta = True
)

docs = retriever.invoke("large language models")

for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content)


    