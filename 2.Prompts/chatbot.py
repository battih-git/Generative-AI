from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

chat_history = [
    SystemMessage(content="Your'e a helpful assistant."),

    
    ]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI: ', result.content)

print(chat_history)