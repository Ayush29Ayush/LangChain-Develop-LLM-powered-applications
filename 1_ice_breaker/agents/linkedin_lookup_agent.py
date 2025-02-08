import sys
import os

# Add parent directory to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from decouple import config  # For secure environment variable handling
from langchain_groq import ChatGroq  # Groq LLM integration
from langchain.prompts.prompt import PromptTemplate  # Template system for prompts
from langchain_core.tools import Tool  # Base class for custom tools
from langchain.agents import (  # Agent creation utilities
    create_react_agent,  # Creates a reactive agent
    AgentExecutor,  # Executes agent actions
)
from tools.tools import get_profile_url_tavily  # Custom tool for LinkedIn searching
from langchain import hub  # Hub for loading predefined prompts


def lookup(name: str) -> str:
    """
    Looks up a person's LinkedIn profile URL using a combination of LLM inference
    and web searching.

    Args:
        name (str): Full name of the person to look up

    Returns:
        str: LinkedIn profile URL
    """
    # Initialize Groq LLM model with temperature=0 for deterministic responses
    llm = ChatGroq(
        model="llama3-8b-8192", temperature=0, api_key=config("GROQ_API_KEY")
    )

    # Define prompt template for LinkedIn URL extraction
    # This template expects a single variable {name_of_person}
    template = """given the full name {name_of_person} I want you to get it me a link
                 to their Linkedin profile page. Your answer should contain only a URL"""

    # Create template instance with variable mapping
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    # Define custom tool for LinkedIn profile searching
    tools_for_agent = [
        Tool(
            name="Crawl Google for linkedin profile page",
            func=get_profile_url_tavily,  # Pass function reference without parentheses
            description="useful for when you need get the Linkedin Page URL",
        )
    ]

    # Load predefined react prompt template
    react_prompt = hub.pull("hwchase17/react")

    # Create agent with LLM, tools, and prompt
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    # Create executor instance for running agent actions
    agent_executor = AgentExecutor(
        agent=agent, tools=tools_for_agent, verbose=True  # Enable detailed logging
    )

    # Execute the agent with formatted prompt
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    # Print result for debugging
    # print("Linkedin Lookup Agent Result =>", result)

    # Extract and return the LinkedIn URL
    linked_profile_url = result["output"]
    return linked_profile_url


if __name__ == "__main__":
    # Example usage with test case
    print(lookup(name="Ayush Senapati Cozentus VIT"))
