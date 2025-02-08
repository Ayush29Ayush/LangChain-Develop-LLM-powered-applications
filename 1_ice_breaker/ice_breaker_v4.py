from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

def ice_break_with(name: str) -> str:
    linkedin_username_link = linkedin_lookup_agent(name=name)
    print("Linkedin Profile URL =>", linkedin_username_link)
    # linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username_link, mock=False)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username_link, mock=True)
    
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
    
    try:
        res = chain.invoke(input={"information": linkedin_data})
        print(res)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Hello LangChain!" + "\n")
    print("--" * 50)
    ice_break_with(name="Ayush Senapati Cozentus VIT")
    print("--" * 50)
    print("\n" + "Bye LangChain!" + "\n")

    
