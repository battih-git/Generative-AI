from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions = 32)
docs = [
    'Delhi is a capital of India',
    'Kolkata is the capital of West Bengal'
]

result = embedding.embed_documents(docs)
print(result)

