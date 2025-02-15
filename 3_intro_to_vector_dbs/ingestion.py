import os
from dotenv import load_dotenv


load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    print("INDEX_NAME:", os.getenv("INDEX_NAME"))
    print("PINECONE_API_KEY:", os.getenv("PINECONE_API_KEY"))