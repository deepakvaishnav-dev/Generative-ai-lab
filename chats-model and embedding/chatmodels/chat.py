from dotenv import load_dotenv

load_dotenv() 

#chatgpt open ai

# from langchain.chat_models import init_chat_model

# model = init_chat_model("gpt-5.5") 

# response =model.invoke("What is the capital of France?") 
# print(response)

#Gemini 2.5 flash

# from langchain.chat_models import init_chat_model

# model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# response = model.invoke("When babar cames in india?")
# print(response.content)

#Second type of usig this model

# from langchain_google_genai import ChatGoogleGenerativeAI

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# response = model.invoke("Give me a paragraph on machine learning")
# print(response.content)

#GROQ using this model 

# from langchain.chat_models import init_chat_model

# model = init_chat_model("groq:openai/gpt-oss-120b")

# response = model.invoke("what is freedom of india?")
# print(response.content)


# Temperature and other parameters can be set when initializing the model. For example:

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0.1, max_output_tokens=200000)

response = model.invoke("Give me a prompt to generate a story about a dragon and a princess.")
print(response.content)