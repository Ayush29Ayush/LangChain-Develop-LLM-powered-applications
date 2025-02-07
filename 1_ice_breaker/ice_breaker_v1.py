from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import openai

information = """
    Elon Reeve Musk is a businessman and U.S. Special Government employee, best known for his key roles in Tesla, Inc., SpaceX, and his ownership of Twitter. Musk is the wealthiest individual in the world; as of January 2025, Forbes estimates his net worth to be US$426 billion. Musk's actions and expressed views have further solidified his status as a public figure.A member of the wealthy South African Musk family, Musk was born in Pretoria before immigrating to Canada, acquiring its citizenship. He moved to California in 1995 to attend Stanford University, and with his brother Kimbal co-founded the software company Zip2, that was later acquired by Compaq in 1999. That same year, Musk co-founded X.com, a direct bank, that later formed PayPal. In 2002, Musk acquired U.S. citizenship, and eBay acquired PayPal. Using the money he made from the sale, Musk founded SpaceX, a spaceflight services company, in 2002. In 2004, Musk was an early investor in electric vehicle manufacturer Tesla and became its chairman and later CEO. In 2018, the U.S. Securities and Exchange Commission (SEC) sued Musk, alleging he falsely announced that he had secured funding for a private takeover of Tesla, stepped down as chairman, and paid a fine. In 2022, he acquired Twitter, and rebranded the service as X the following year. In January 2025, Musk was appointed director of the Department of Government Efficiency as a special government employee. 
"""

if __name__ == "__main__":
    print("Hello LangChain!")

    summary_template = """
        Given {information} about a person, create:

        1. A 2-3 sentence summary including:
            - Current role, key achievements, and current status.
        2. Two verified facts with dates/details.

        Format with markdown headers and bullet points:
        ## Summary
        ## Interesting Facts
        1. [Fact 1]
        2. [Fact 2]

        Ensure professional tone and factual accuracy. 
    """


    summary_prompt_template = PromptTemplate(
        template=summary_template, input_variables=["information"]
    )

    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=config("OPENAI_API_KEY"))
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=config("GROQ_API_KEY"))

    chain = summary_prompt_template | llm

    #! For OpenAI
    # try:
    #     res = chain.invoke(input={"information": information})
    #     print(res)
    # except openai.RateLimitError:
    #     print("Rate limit exceeded. Please wait and try again later.")
        
    #! For Groq
    try:
        res = chain.invoke(input={"information": information})
        print(res.content)
    except Exception as e:  # Catching general exception
        print(f"Error: {str(e)}")
        if "rate limit" in str(e).lower():
            print("Rate limit exceeded. Please wait and try again later.")
