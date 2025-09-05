import asyncio
from pydantic import BaseModel
from config import client, model_name  # make sure you have your OpenAI client configured
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
)

# -------------------
# Disable tracing
# -------------------
set_tracing_disabled(disabled=True)

# -------------------
# Formatter
# -------------------
class DotFormatter:
    @staticmethod
    def format_list(items: list[str]) -> str:
        return "\n".join([f". **{item}**" for item in items])

    @staticmethod
    def format_numbered(items: list[str]) -> str:
        return "\n".join([f"{i+1}. **{item}**" for i, item in enumerate(items)])

# -------------------
# Output Schema
# -------------------
class MessageOutput(BaseModel):
    response: str

# -------------------
# Core Dental Content
# -------------------
CORE_CONTENT = {
    "conditions": {
        "gum infection": {
            "purpose": "Detect gum inflammation or gingivitis",
            "how_to_use": "Upload a photo of your gums and answer quiz questions",
            "benefits": "Helps early detection and prevents progression",
        },
        "tooth decay": {
            "purpose": "Identify cavities or decay",
            "how_to_use": "Upload a tooth photo, select pain area, answer sensitivity quiz",
            "benefits": "Provides early treatment recommendations",
        },
        "sensitivity": {
            "purpose": "Detect enamel thinning or exposed roots",
            "how_to_use": "Describe your symptoms (cold/hot/chewing pain)",
            "benefits": "Guides next steps for treatment",
        },
    },
    "overview": [
        "Smart Dental Assistant: Upload a photo + symptoms to get a diagnosis",
        "Symptom Quiz: Guided questions about your pain",
        "Voice Assistant: Ask about dental health",
        "Doctor Connect: Schedule appointments after diagnosis",
    ],
    "faqs": {
        "what is this app": "It’s an AI-powered dental health checker that provides initial assessments.",
        "how do i use it": "Take a clear photo of your teeth, upload it, and answer guided questions.",
        "is this a replacement for a dentist": "No, it provides initial advice only. You should always consult a licensed dentist.",
    }
}

# -------------------
# Tools
# -------------------
@function_tool
def get_condition_info(condition_name: str) -> str:
    conditions = CORE_CONTENT["conditions"]
    condition = conditions.get(condition_name.lower())
    if not condition:
        return DotFormatter.format_list([
            f"I only know these conditions: {', '.join(conditions.keys())}",
            "Pick one!"
        ])
    return DotFormatter.format_list([
        f"{condition_name.title()}",
        f"Purpose: {condition['purpose']}",
        f"How to use: {condition['how_to_use']}",
        f"Benefits: {condition['benefits']}"
    ])

@function_tool
def get_overview() -> str:
    return DotFormatter.format_numbered(CORE_CONTENT["overview"])

@function_tool
def list_conditions() -> str:
    return DotFormatter.format_list(["Conditions I can check:"] + list(CORE_CONTENT["conditions"].keys()))

@function_tool
def answer_faq(question: str) -> str:
    faqs = CORE_CONTENT["faqs"]
    q = question.lower().strip()
    for k, v in faqs.items():
        if k in q:
            return DotFormatter.format_list([v])
    return DotFormatter.format_list([
        "I only answer questions about dental conditions and app usage.",
        "Try asking about gum infection, tooth decay, or sensitivity."
    ])

# -------------------
# Dental AI Agent
# -------------------
dental_agent = Agent(
    name="Dental Diagnosis Agent",
    instructions=(
        "You are a dental AI assistant.\n"
        "Analyze user text and uploaded dental photos for possible issues.\n"
        "Never give final medical advice—always recommend seeing a dentist for confirmation.\n"
        "Stick to gum infection, tooth decay, and sensitivity.\n"
        "Use dot or numbered list style for clarity."
    ),
    model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
    tools=[get_condition_info, get_overview, list_conditions, answer_faq],
    output_type=MessageOutput,
)

# -------------------
# Runner
# -------------------
async def run_dental_agent(query: str) -> str:
    try:
        result = await Runner.run(dental_agent, query)
        response = result.final_output.response
        if not response.strip():
            return get_overview()
        return response
    except Exception:
        return get_overview()

# -------------------
# Image Analysis Helper
# -------------------
async def analyze_dental_image(image_url: str, symptoms: str = "") -> str:
    response = client.chat.completions.create(
        model="gpt-4o",  # Vision model
        messages=[
            {"role": "system", "content": "You are a dental AI assistant. Analyze teeth images for possible issues."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Patient symptoms: {symptoms}"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}
        ]
    )
    return response.choices[0].message["content"]

# -------------------
# Terminal test
# -------------------
if __name__ == "__main__":
    async def main():
        print(". **Welcome! Ask me about dental conditions or upload an image for diagnosis.**")
        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                print(". **Bye!**")
                break
            response = await run_dental_agent(query)
            print(response + "\n")

    asyncio.run(main())
