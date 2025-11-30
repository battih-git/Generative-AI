from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id ='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)
# 1st prompt -> detailed report
template_1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables =['topic']
)

# 2nd prompt

template_2 = PromptTemplate(
    template="Write a 5 line summary on following text. /n{text}",
    input_variables =['text']
)


prompt1 = template_1.invoke({'topic':'black hole'})
result = model.invoke(prompt1)

prompt2 = template_2.invoke({'text':result.content})
result1 = model.invoke(prompt2)

print(result1.content)


parser = StrOutputParser()

chain = template_1 | model | parser | template_2 | model | parser
result = chain.invoke({'topic':'black hole'})
print(result)