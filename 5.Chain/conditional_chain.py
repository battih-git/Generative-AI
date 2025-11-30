from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts  import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Feedback(BaseModel):
    sentiment: Literal['positive','negative']=Field(description="Give the sentiment of the feedback")

parser = StrOutputParser()

parser1 = PydanticOutputParser(pydantic_object=Feedback)

prompt_1 = PromptTemplate(
    template= "Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction' : parser1.get_format_instructions}
)

classifier_chain = prompt_1 | model | parser1

prompt_2 = PromptTemplate(
    template = "Write an appropriate response to this negative feedback \n{feedback}",
    input_variables=['feedback']
)

prompt_3 = PromptTemplate(
    template = "Write an appropriate response to this positive feedback \n{feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt_3 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt_2 | model | parser),
    RunnableLambda(lambda x: 'could not find sentiment'), 
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback':'This is the beautiful phone'}))