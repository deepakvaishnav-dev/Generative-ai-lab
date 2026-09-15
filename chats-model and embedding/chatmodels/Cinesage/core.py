from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class MovieInfo(BaseModel):
    title: str
    release_year : Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=MovieInfo)

prompt = ChatPromptTemplate.from_messages([
    ('system', """Extract movie information from the paragraph.
{format_instructions}
"""),
    ("human", "{paragraph}")
])

print("Enter the movie paragraph (paste your text, then press Enter on an empty line to finish):\n")
lines = []
while True:
    try:
        line = input()
    except EOFError:
        break
    if line.strip() == "" and lines:
        break
    lines.append(line)

paragraph = "\n".join(lines).strip()

if not paragraph:
    print("No paragraph provided. Exiting.")
    exit(0)

final_prompt = prompt.invoke(
    {
        "paragraph": paragraph,
        "format_instructions": parser.get_format_instructions()
    }
)

response = model.invoke(final_prompt)
parsed_output: MovieInfo = parser.invoke(response)

print("\n--- Parsed Movie Information ---")
print(f"Title: {parsed_output.title}")
print(f"Release Year: {parsed_output.release_year}")
print(f"Genre: {', '.join(parsed_output.genre)}")
print(f"Director: {parsed_output.director}")
print(f"Cast: {', '.join(parsed_output.cast)}")
print(f"Rating: {parsed_output.rating}")
print(f"Summary: {parsed_output.summary}")