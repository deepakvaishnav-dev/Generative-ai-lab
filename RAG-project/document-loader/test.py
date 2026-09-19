from langchain_community.document_loaders import TextLoader

data = TextLoader("note.txt", encoding="utf-8")

docs = data.load()

print(docs)

