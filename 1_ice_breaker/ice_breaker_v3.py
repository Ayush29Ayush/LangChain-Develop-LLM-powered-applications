from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    print("Hello LangChain!")

    summary_template = """
        Based on the LinkedIn information provided about the individual: {information}, please generate the following:

        1. A concise professional summary.
        2. A detailed overview of their current job role, 
        3. A detailed overview of their key achievements and professional status.
        4. Two intriguing and unique facts about the individual.

        Thank you!
    """

    summary_prompt_template = PromptTemplate(
        template=summary_template, input_variables=["information"]
    )

    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=config("GROQ_API_KEY"))

    chain = summary_prompt_template | llm | StrOutputParser()
    
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/ayush-senapati-a531b8145/", mock=True)

    try:
        res = chain.invoke(input={"information": linkedin_data})
        print(res)
    except Exception as e:
        print(f"Error: {str(e)}")
