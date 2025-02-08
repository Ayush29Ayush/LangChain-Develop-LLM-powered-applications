from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import summary_parser

def ice_break_with(name: str) -> str:
    """
    Creates an engaging icebreaker summary from LinkedIn profile data using AI.
    
    Args:
        name (str): Full name of the person to create icebreaker for
        
    Returns:
        str: Structured summary containing professional information and interesting facts
    """
    
    # First, get the LinkedIn profile URL using our lookup agent
    linkedin_username_link = linkedin_lookup_agent(name=name)
    print("Linkedin Profile URL =>", linkedin_username_link)
    
    # Scrape the LinkedIn profile data (using mock data for testing)
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_username_link, 
        mock=True
    )
    
    # Define template for generating professional summary
    summary_template = """
    Based on the LinkedIn information provided about the individual: {information}, please generate the following:
    1. A concise professional summary.
    2. A detailed overview of their current job role,
    3. A detailed overview of their key achievements and professional status.
    4. Two intriguing and unique facts about the individual.
    Thank you!
    \n {format_instructions}
    """
    
    # Create prompt template with format instructions for structured output
    summary_prompt_template = PromptTemplate(
        template=summary_template, 
        input_variables=["information"], 
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
    )
    
    # Initialize Groq LLM with deterministic responses
    llm = ChatGroq(
        model="llama3-8b-8192", 
        temperature=0, 
        api_key=config("GROQ_API_KEY")
    )
    
    # Create processing chain: template -> LLM -> string parser -> summary parser
    chain = summary_prompt_template | llm | StrOutputParser() | summary_parser
    
    try:
        # Execute the chain with LinkedIn data
        res = chain.invoke(input={"information": linkedin_data})
        print("res =>", res)
        print("Type of res =>", type(res), "\n")
        # Convert structured output to dictionary/json format
        res_dict = res.to_dict()
        print("res_dict =>", res_dict)
        print("Type of res_dict =>", type(res_dict))
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Hello LangChain!" + "\n")
    print("--" * 50)
    ice_break_with(name="Ayush Senapati Cozentus VIT")
    print("--" * 50)
    print("\n" + "Bye LangChain!" + "\n")