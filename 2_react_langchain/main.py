from typing import List, Union
from langchain.agents import tool
from langchain.prompts.prompt import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_groq import ChatGroq
from decouple import config
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.agents.format_scratchpad import format_log_to_str

@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the text by characters.
    """
    print(f"get_text_length enter with text: {text}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool {tool_name} not found in tools: {tools}")


if __name__ == "__main__":
    print("Hello ReAct LangChain!" + "\n")
    # print(get_text_length(text="Human")) #! This is used to call the function when it is not used as a tool
    # print(get_text_length.invoke(input={"text": "Human"})) #! This is used to call the function when it is used as a tool
    tools = [get_text_length]

    template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
    """

    # prompt = PromptTemplate.from_template(template=template).partial(tools=tools, tool_names=[tool.name for tool in tools])
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), 
        tool_names=", ".join([tool.name for tool in tools])
    )

    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        api_key=config("GROQ_API_KEY"),
        stop_sequences=["\nObservation", "Observation"]
    )
    intermediate_steps = []

    # print(prompt)
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # res = agent.invoke({"input": "What is the length of 'Human' in characters?"})
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length in characters of text: Human?",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)

    if isinstance(agent_step, AgentFinish):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools=tools, tool_name=tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"Observation: {observation}")
        intermediate_steps.append((agent_step, str(observation)))
        
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length in characters of text: Human?",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)
    
    if isinstance(agent_step, AgentFinish):
        print("### AgentFinish ###")
        print(agent_step.return_values)