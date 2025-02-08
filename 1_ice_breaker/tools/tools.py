from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from decouple import config

def get_profile_url_tavily(name: str):
    """
    Searches for a person's LinkedIn or Twitter profile URL using Tavily Search.
    
    Args:
        name (str): Full name of the person to search for
        
    Returns:
        str: URL of the matching social media profile
    """
        
    # Retrieve Tavily API key from environment configuration
    tavily_api_key = config("TAVILY_API_KEY")
    
    # Initialize API wrapper with authentication credentials
    api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
    
    # Create search client with authenticated wrapper
    search = TavilySearchResults(api_wrapper=api_wrapper)
    
    # Execute search with formatted query string
    res = search.run(f"{name}")
    
    # Extract and return the first matching URL
    return res[0]["url"]