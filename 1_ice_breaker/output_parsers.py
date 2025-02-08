from typing import List, Dict, Any
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Define a data model to structure information about a person
class Summary(BaseModel):
    # Main summary text about the person
    summary: str = Field(description="summary")
    
    # Collection of interesting facts about the person
    facts: List[str] = Field(description="interesting facts about them")
    
    # Convert the model instance to a dictionary format
    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "facts": self.facts}

# Create a parser that will convert LLM output into our structured Summary model
summary_parser = PydanticOutputParser(pydantic_object=Summary)