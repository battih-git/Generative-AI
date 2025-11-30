from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch

def word_counter(text):
    return len(text.split())


load_dotenv()

model = GoogleGenerativeAI(model='gemini-2.5-flash')


prompt = PromptTemplate(
    template='Write a detailed report on - {topic}',
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template='SUmmarize the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt,model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':"Russia v/s Ukraine"}))