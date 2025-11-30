from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

model = GoogleGenerativeAI(model='gemini-2.5-flash')

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template='Generate a linkedin post about {topic}',
    input_variables= ['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet':RunnableSequence(prompt1, model, parser),
        'linkedin':RunnableSequence(prompt2, model, parser)
    }
)

result = parallel_chain.invoke({'topic':'AI'})
print(result)