from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()
model = GoogleGenerativeAI(model='gemini-2.5-flash')

messages = [
    SystemMessage(content="You're a helpful assistant."),
    HumanMessage(content="Tell me about a langchain")
]

messages.append(AIMessage(model.invoke(messages)))

print(messages[-1])