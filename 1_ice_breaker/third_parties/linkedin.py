import os
import requests
from decouple import config

def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = True):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""

    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/Ayush29Ayush/c42be02c67249609306805ccf8434e2a/raw/de55b3daad3bd36130f1250eb5dc507f8f444edf/ayush-senapati-linkedin-scraping.json"
        response = requests.get(linkedin_profile_url,timeout=10)
    else:
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "apikey": config("SCRAPIN_API_KEY"),
            "linkedInUrl": linkedin_profile_url,
        }
        response = requests.get(api_endpoint,params=params,timeout=10)

    data = response.json().get("person")

    filtered_data = {}

    for key, value in data.items():
        if value not in ([], "", "", None) and key != "certifications":
            filtered_data[key] = value

    # print("Filtered Data =>", filtered_data)
    return filtered_data

if __name__ == "__main__":
    print(scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/ayush-senapati-a531b8145/",mock=True))
