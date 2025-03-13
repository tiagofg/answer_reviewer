from pydantic import BaseModel
from typing import Dict, Any

# Input model: expects a JSON with the fields "question", "answer", and "context"
class RevisionRequest(BaseModel):
    question: str
    answer: str
    context: Dict[str, Any]
