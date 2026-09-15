from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Define the remote inference endpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3,
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Give me a prompt to generate a story about a dragon and a princess.")
print(response.content)
