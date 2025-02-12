from langchain.agents import tool
from langchain.prompts.prompt import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_groq import ChatGroq
from decouple import config


@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the text by characters.
    """
    print(f"get_text_length enter with text: {text}")
    text = text.strip("'\n").strip('"')
    return len(text)


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
        Thought:
    """

    # prompt = PromptTemplate.from_template(template=template).partial(tools=tools, tool_names=[tool.name for tool in tools])
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), tool_names=[tool.name for tool in tools]
    )

    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        api_key=config("GROQ_API_KEY"),
        stop_sequences=["\nObservation"],
    )
    
    print(prompt)
