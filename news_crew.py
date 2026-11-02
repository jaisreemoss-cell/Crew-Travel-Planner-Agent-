import os
import sys
from datetime import date

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# ---------------------------------------------------------------
# 1. Load both API keys from .env (never hard-code them)
# ---------------------------------------------------------------
load_dotenv(override=True)

for key in ("OPENAI_API_KEY", "SERPER_API_KEY"):
    if not os.getenv(key):
        sys.exit(f"{key} is not set. Add it to your .env file and try again.")

today = date.today().strftime("%d %B %Y")  # e.g. 02 November 2026

# ---------------------------------------------------------------
# 2. The model both agents think with, and the search tool
# ---------------------------------------------------------------
# MODEL comes from your .env (set on Day 3); gpt-4o-mini is the fallback.
llm = LLM(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0.3)

# SerperDevTool reads SERPER_API_KEY from the environment by itself.
# search_type="news" makes it search Google News instead of normal results.
search = SerperDevTool(search_type="news", n_results=10)

# ---------------------------------------------------------------
# 3. The two agents
# ---------------------------------------------------------------
researcher = Agent(
    role="Tech News Researcher",
    goal="Find the 3 most important technology news stories published today, {today}.",
    backstory=(
        "You are a careful tech journalist who scans the news every morning. "
        "You only report stories you actually found in your search results, "
        "you prefer reputable sources, and you never make up headlines, facts or links."
    ),
    tools=[search],  # only the Researcher can search
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Newsletter Writer",
    goal="Turn the researcher's 3 stories into a short, clear 5-line newsletter.",
    backstory=(
        "You write a daily tech newsletter for busy professionals who read it "
        "in under a minute. You write in plain, friendly English, you keep every "
        "line short, and you only use the facts you are given."
    ),
    # no tools: the Writer must write, not research again
    llm=llm,
    verbose=True,
)

# ---------------------------------------------------------------
# 4. The two tasks
# ---------------------------------------------------------------
find = Task(
    description=(
        "Search the web for today's top technology news ({today}). "
        "Pick the 3 most important stories from the last 24-48 hours, "
        "each about a different topic or company."
    ),
    expected_output=(
        "A numbered list of exactly 3 stories, nothing else. For each story:\n"
        "Headline: <headline>\n"
        "Source: <publication name>\n"
        "Summary: <1-2 sentences on what happened and why it matters>\n"
        "Link: <URL from the search results>"
    ),
    agent=researcher,
)

write = Task(
    description=(
        "Using ONLY the 3 stories from the research task, write a tech newsletter "
        "for {today}. Do not add any story or fact that is not in the research."
        "If the research does not contain 3 real stories with sources, "
    "output only this line: 'No verified tech news available today - research step failed.'"
    ),
    expected_output=(
        "Exactly 5 lines of plain text, nothing before or after them:\n"
        "Line 1: a short newsletter title that includes the date.\n"
        "Lines 2-4: one line per story, in the format "
        "'<number>. <Headline> - <one-sentence takeaway> (<Source>)'.\n"
        "Line 5: a one-sentence closing thought linking the stories.\n"
        "Each line under 30 words. No blank lines, no extra notes."
    ),
    agent=writer,
    context=[find],  # hand the Researcher's output to the Writer
)

# ---------------------------------------------------------------
# 5. Put them in a crew and run them in order
# ---------------------------------------------------------------
crew = Crew(
    agents=[researcher, writer],
    tasks=[find, write],
    process=Process.sequential,  # find runs first, then write
    verbose=True,
)

result = crew.kickoff(inputs={"today": today})

print("\n========== TODAY'S TECH NEWSLETTER ==========\n")
print(result)