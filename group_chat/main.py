from fastapi import FastAPI, HTTPException
import uvicorn
from typing import List

# Import the model and service
from models.revision import RevisionRequest
from services.revision_service import RevisionService


app = FastAPI(
    title="Response Revision API",
    description=(
        "Receives a question via POST containing the fields 'question', 'answer', and 'context', "
        "sends it to the initiate_chat method for evaluation, and returns only the final answer."
    ),
    version="1.4.0"
)

# Create an instance of the revision service
revision_service = RevisionService()

@app.post("/revise")
def revise_question(request: RevisionRequest):
    try:
        response = revision_service.process_revision(request)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/revise-questions")
def revise_questions(requests: List[RevisionRequest]):
    try:
        responses = revision_service.process_revisions(requests)
        
        return {"responses": responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
