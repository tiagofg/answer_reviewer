from pydantic import BaseModel
from typing import Dict, Any, List

class RevisionRequest(BaseModel):
    id: int
    question: str
    answer: str
    correct: bool
    feedback: str | None
    locale: str
    intent: Dict[str, Any]
    context: Dict[str, Any]
    # metadata: List[Any]
    category: str
