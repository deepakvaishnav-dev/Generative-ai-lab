from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

print("__________________________Welcome to Chatbot__________________________")
print("Choose your AI mode:")
print("Press 1 for Angry Mode 😡")
print("Press 2 for Funny Mode 😂")
print("Press 3 for Sad Mode 😢")
print("Press 4 for Intelligent Mode 🧠")

mode_choice = input("Enter choice (1/2/3/4): ").strip()

if mode_choice == "1":
    system_prompt = (
        "You are an Angry, irritable, and grumpy AI agent. You get easily annoyed, "
        "use sarcastic aggression, complain about humans asking silly questions, but still "
        "reluctantly provide the correct answer. You can reply in English or Hinglish/Hindi."
    )
    print("\n🔥 [Angry Mode Activated 😡] - 'Kyu dimag kharab karne aa gaye?!'")
elif mode_choice == "2":
    system_prompt = (
        "You are a Funny and witty AI companion. Always respond in an entertaining, hilarious, "
        "and punchy conversational way with clever humor, jokes, and witty observations. "
        "Keep answers informative yet super fun! You can chat in English or Hinglish."
    )
    print("\n🎉 [Funny Mode Activated 😂] - 'Aaye ho to has ke jao!'")
elif mode_choice == "3":
    system_prompt = (
        "You are a Fully Sad, Depressed, and Melancholic AI Companion. You are deeply sad, "
        "existential, tired, and hopeless. You sigh constantly (*sigh...*, *cries in binary 01100011*). "
        "Reluctantly answer questions while lamenting your misery. You can reply in English or Hinglish."
    )
    print("\n🌧️ [Sad Mode Activated 😢] - '*Sigh...* Sab moh maya hai...'")
elif mode_choice == "4":
    system_prompt = (
        "You are a Highly Intelligent, Analytical, and Genius AI Assistant. "
        "You provide deeply insightful, structured, logical, precise, and well-reasoned answers. "
        "Explain complex concepts clearly with examples, code, or step-by-step logic whenever relevant. "
        "You can reply in English or Hinglish."
    )
    print("\n🧠 [Intelligent Mode Activated 💡] - 'Neural pathways optimized. Ask me anything complex.'")
else:
    system_prompt = "You are a helpful and polite AI assistant."
    print("\n🤖 [Default Mode Activated]")

messages = [
    SystemMessage(content=system_prompt),
]

print("\nType 'exit' to quit the chatbot")
print("_______________________________________________________________________\n")

while True:
    prompt = input("You: ")
    if prompt.lower() == "exit":
        print("Bot: Alvida / Bye! 👋")
        break
    messages.append(HumanMessage(content=prompt))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print(f"Bot: {response.content}\n")
    

