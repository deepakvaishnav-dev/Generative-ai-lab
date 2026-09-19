import os
import sys
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

file_path = os.path.join(os.path.dirname(__file__), "document-loader", "note.txt")
data = TextLoader(file_path, encoding="utf-8")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system","You are a AI that summarizes the text"),
    ("human","{data}")]
)


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = template.format_messages(data = docs[0].page_content)

result = model.invoke(prompt)

print(result.content)