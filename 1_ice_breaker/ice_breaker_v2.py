from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

information = """
    Pizza without cheese
"""

if __name__ == "__main__":
    print("Hello LangChain!" + "\n")

    summary_template = """
        Write a very short song on the topic : {information}
    """


    summary_prompt_template = PromptTemplate(
        template=summary_template, input_variables=["information"]
    )

    llm = ChatOllama(model="llama3", temperature=0)

    # chain = summary_prompt_template | llm 
    chain = summary_prompt_template | llm | StrOutputParser()
    
    #! For Ollama
    try:
        res = chain.invoke(input={"information": information})
        # print(res.content)
        print(res)
    except Exception as e:
        print(f"Error: {str(e)}")
