from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = GoogleGenerativeAI(model='gemini-2.5-flash')

prompt = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

prompt_2 = PromptTemplate(
    template="Explain this joke - {text}",
    input_variables=['text']
)

chain = RunnableSequence(prompt,model,parser, prompt_2, model, parser)

print(chain.invoke({'topic':"AI"}))