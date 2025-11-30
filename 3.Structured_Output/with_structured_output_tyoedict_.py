from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Schema
class Review(TypedDict):
    key_themes: Annotated [list[str], 'Write down all the key themes discussed in the review in a list.']
    summary: Annotated[str,"A bried summary of the review"]
    sentiment: Annotated[Literal['pos','neg'], 'Return sentiment of the review either Positive or Negative']
    pros: Annotated[Optional[list[str]], 'Write down all the pros inside a list']
    cons: Annotated[Optional[list[str]], 'Write down all the cons inside a list']

parser = JsonOutputParser(pydantic_object=None, json_schema=Review)

prompt = PromptTemplate(
    template="""
Extract structured info from the text.

Return a JSON object with:
- summary: short summary
- sentiment: positive/negative/neutral

Text:
{review}

{format_instructions}
""",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    convert_to_openai_format=True  # important!
)

chain = prompt | model | parser

result = chain.invoke({
    "review": """
The Analogue 3D is finally here, and it's one of the best retro games consoles available on the market. After multiple delays and what felt like a longer wait than those excruciating Christmas Eve sleeps as a kid, Analogue’s take on the iconic Nintendo 64 is in my hands, and I’ve been testing it thoroughly over the last couple of weeks.

The Analogue 3D is a modern take on the N64, allowing gamers to experience the magic and nostalgia of the home console that launched in 1996. Analogue has made a name for itself as one of, if not the, best retro game preservation company on the planet, and I don’t think it will take long for the 3D to cement itself as the definitive N64 experience, just like the company’s Game Boy hardware emulator, the Analogue Pocket.

Analogue’s approach to hardware is one of true love for nostalgia, allowing gamers to recreate the memories of their youth by playing original cartridges in 4K via an FPGA chip. This means the 3D is for the true enthusiast, someone who has a collection of N64 games and wants to experience their childhood in the best way possible.

Build quality is excellent, as you’d expect from the company that created the Analogue Pocket, and the console works with original N64 controllers as well as 8bitdo’s modern recreation, which has been purposefully built for use with the 3D.

The Analogue 3D is the quintessential Nintendo 64 experience, and is a must-buy for those looking to play the most authentic recreation of the console of their youth on one of the best OLED TVs. That said, it’s still an N64, so if you don’t have a physical collection or don’t have the necessary nostalgia to truly enjoy these often dated games, you may want to opt for a different era of retro gaming instead.

The Analogue 3D is now shipping via analogue.co. The 3D first went on preorder in October 2024, and while it’s now available, it is currently sold out on the Analogue website.

You can buy the 3D in Black or White, and in the box you’ll get an HDMI cable, USB-C for power, and a power adaptor. The 3D has no controller in the box, and while you can use any of your old N64 controllers, third-party company 8bitdo’s 64 controller is available for $39.99 / £32 in black and white via Amazon.

"""
})

print(result)
print(type(result))
