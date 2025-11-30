from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

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

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)