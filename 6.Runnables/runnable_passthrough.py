from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()

model = GoogleGenerativeAI(model='gemini-2.5-flash')

passthrough = RunnablePassthrough()
passthrough.invoke(2)


prompt1 = PromptTemplate(
    template='Write one joke about - {topic}',
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template='Explain the joke {text}',
    input_variables= ['text']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'Cricket'}))
