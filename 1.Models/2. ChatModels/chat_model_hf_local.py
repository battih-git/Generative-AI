from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()

# Create pipeline with CORRECT parameter names
pipe = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",
    max_new_tokens=100,  # Fixed: 'max_new_tokens' not 'max_new_token'
    temperature=0.7,     # Fixed: 'temperature' not 'temprature'
    device=-1
)

# Create LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create chat model
model = ChatHuggingFace(llm=llm)

# Add question mark for proper sentence
result = model.invoke("What is the capital of India?")
print(result.content)