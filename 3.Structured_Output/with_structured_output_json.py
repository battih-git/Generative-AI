from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --- Schema ---
class Review(BaseModel):
    key_themes: list[str] = Field(description="Key themes discussed in the review.")
    summary: str = Field(description="A brief summary of the review.")
    sentiment: Literal['pos', 'neg'] = Field(description="Sentiment of the review (pos or neg).")

# --- Parser ---
parser = PydanticOutputParser(pydantic_object=Review)

# --- Prompt ---
prompt = PromptTemplate(
    template="""
Extract structured information from the review.

Return ONLY valid JSON that matches this schema:
{format_instructions}

Review:
{review}
""",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# --- Model ---
model = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    convert_to_openai_format=True
)

# --- Chain ---
chain = prompt | model | parser

# --- Invoke ---
result: Review = chain.invoke({
    "review": "The hardware is great but the software feels bloated…"
})

# Print as Python dict
print(result.dict())

# Print as formatted JSON
print(result.json(indent=2))
