import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv
import json
import uvicorn
from agents import reviewer, user_proxy  # Import agents from the agents module

load_dotenv()

app = FastAPI(
    title="Response Revision API",
    description=(
        "Receives a question via POST containing the fields 'question', 'answer' and 'context', "
        "sends it to the initiate_chat method for evaluation and returns only the final answer."
    ),
    version="1.4.0"
)

# Input model: expects a JSON with the fields "question", "answer" and "context"
class RevisionRequest(BaseModel):
    question: str
    answer: str
    context: Dict[str, Any]

@app.post("/revise")
def revise_question(request: RevisionRequest):
    try:
        # Construct the complete JSON with the three fields
        question_data = {
            "question": request.question,
            "answer": request.answer,
            "context": request.context
        }

        formatted_question = json.dumps(question_data, indent=2, ensure_ascii=False)
        message = f"Please evaluate the following answer:\n{formatted_question}"

        # Initiate the chat for evaluation/revision
        result = user_proxy.initiate_chat(reviewer, message=message)

        # Attempt to extract the final answer from the last message sent by user_proxy in the chat_history
        final_answer = request.answer
        if hasattr(result, "chat_history") and isinstance(result.chat_history, list):
            # Search for the last message sent by user_proxy (role "assistant" with name "User")
            for msg in reversed(result.chat_history):
                if msg.get("role") == "assistant" and msg.get("name") == "User":
                    if "Revised Answer:" in msg.get("content"):
                        final_answer = msg.get("content").split("Revised Answer:")[1].strip().replace('"', '')
                    elif "It is not possible to provide a revised answer." in msg.get("content"):
                        final_answer = "It is not possible to provide a revised answer."

                    break

        # Extract the total cost from the result object
        total_cost = result.cost.get('usage_including_cached_inference', {}).get('total_cost')
        if total_cost is None:
            total_cost = result.cost.get('usage_excluding_cached_inference', {}).get('total_cost')
        
        if total_cost is not None:
            cost_str = f"${total_cost}"
        else:
            cost_str = "Cost information not available"

        # Determine the Revised Answer field:
        # If final_answer is equal to the original answer, it should be "-"
        # If final_answer is "It is not possible to provide a revised answer.", then it should be null (None in Python, convertido para null no JSON)
        if final_answer == request.answer:
            revised_answer = "-"
        elif final_answer == "It is not possible to provide a revised answer.":
            revised_answer = None
        else:
            revised_answer = final_answer

        # Save the result in "results.json"
        results_file = "results.json"
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                try:
                    results_data = json.load(f)
                except Exception:
                    results_data = []
        else:
            results_data = []

        new_record = {
            "Question": request.question,
            "Original Answer": request.answer,
            "Revised Answer": revised_answer,
            "Cost": cost_str
        }
        results_data.append(new_record)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        return {"response": final_answer.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
