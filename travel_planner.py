

import os
import sys

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# Load OPENAI_API_KEY from .env into the environment (never hard-code it).
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set. Add it to your .env file and try again.")

# The language model the agent thinks with. CrewAI picks up OPENAI_API_KEY
# from the environment automatically, so the key is not passed here.
llm = LLM(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.3,  # low = focused, consistent suggestions
)

# Ask for the city, and ask again if the user just presses Enter.
city = input("Which city? ").strip()
while not city:
    city = input("Please type a city name: ").strip()

planner = Agent(
    role="Local Travel Planner",
    goal=(
        "Recommend the three places in {city} that a first-time visitor "
        "should not miss, each with one short, practical tip."
    ),
    backstory=(
        "You are a seasoned travel planner who has guided thousands of "
        "first-time visitors through cities around the world. Your travellers "
        "usually have only a day or two, so you pick well-known, genuinely "
        "worthwhile landmarks over obscure spots, and you never invent places "
        "that do not exist. You write the way a friend texts advice: short, "
        "specific, and useful, with a tip such as the best time to go."
    ),
    llm=llm,
    verbose=True,
)

task = Task(
    description=(
        "Suggest exactly 3 places to visit in {city} for a first-time visitor "
        "with limited time. Choose real, well-known places located in {city} "
        "itself, and give a mix of experiences where possible (for example, "
        "history, culture, and a viewpoint or market). For each place, say in "
        "a few words what makes it special and add one practical tip."
    ),
    expected_output=(
        "Exactly 3 lines and nothing else: no title, introduction, or closing "
        "remarks. Each line follows this format:\n"
        "<number>. <Place name> - <what makes it special>, <one practical tip>\n"
        "Keep each line under 20 words. Example:\n"
        "1. Amber Fort - hilltop fort with mirrored halls, go early"
    ),
    agent=planner,
)

crew = Crew(agents=[planner], tasks=[task])

result = crew.kickoff(inputs={"city": city})

print(f"\nTop 3 places to visit in {city}:\n")
print(result)