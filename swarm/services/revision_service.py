import os
import csv
import json

from typing import List

from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.multi_agent_chat import initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern

from models.revision import RevisionRequest
from agents.agents import semantic_reviewer, contextual_reviewer, suggester, rewriter, decider, user_proxy

context_variables: ContextVariables = ContextVariables(data={})

class RevisionService:
    def __init__(self, results_file: str = "results.csv"):
        self.results_file = results_file

    def process_revision(self, request: RevisionRequest) -> str:
        global context_variables

        language = "portuguese" if request.locale == "pt" else "spanish"
        intent = request.intent.get("name")

        question_data = {
            "question": request.question,
            "context": request.context,
            "category": request.category,
            "metadata": request.metadata,
            "language": language,
            "intent": intent,
            "original_answer": request.answer,
        }

        formatted_question = json.dumps(question_data, indent=2, ensure_ascii=False)

        context_variables.clear()
        context_variables.update({
            "question": request.question,
            "context": request.context,
            "category": request.category,
            "metadata": request.metadata,
            "language": language,
            "intent": intent,
            "original_answer": request.answer,
            "semantic_score": None,
            "justification_semantic": None,
            "contextual_score": None,
            "justification_contextual": None,
            "original_score": None,
            "suggestions": None,
            "revised_answer": None,
            "revised_answer_semantic_score": None,
            "revised_answer_justification_semantic": None,
            "revised_answer_contextual_score": None,
            "revised_answer_justification_contextual": None,
            "new_score": None,
            "final_answer": None,
            "decision": None,
            "decision_justification": None,
            "number_of_revisions": 0,
        })

        swarm_pattern = DefaultPattern(
            agents=[semantic_reviewer, contextual_reviewer, suggester, rewriter, decider],
            initial_agent=semantic_reviewer,
            context_variables=context_variables,
            user_agent=user_proxy,
        )

        result, final_context, last_agent = initiate_group_chat(
            pattern=swarm_pattern,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "The agents need to work together to review the answer to the question. \n"
                        "If they don't think that the answer is good enough, they should suggest a better one or decide to not answer. \n"
                        "This is the data they have to work with: \n"
                        f"{formatted_question} "
                    )
                }
            ],
            max_rounds=30,
        )

        final_answer = final_context.get("final_answer")
        previous_score = final_context.get("original_score")
        new_score = final_context.get("new_score")
        suggestions = final_context.get("suggestions")
        revised_answer = final_context.get("revised_answer")
        decision = final_context.get("decision")
        decision_justification = final_context.get("decision_justification")
        number_of_revisions = final_context.get("number_of_revisions")

        if (new_score is not None) and (new_score <= 7):
            final_answer = "DO_NOT_ANSWER"

        if (previous_score is not None) and (previous_score > 7):
            final_answer = request.answer

        new_score = new_score if new_score is not None else "-"
        final_answer = final_answer if final_answer != "DO_NOT_ANSWER" else "-"

        # # Extract total cost, if available
        # total_cost = chat_history.cost.get(
        #     'usage_excluding_cached_inference', {}).get('total_cost')
        # cost_str = f"${total_cost}" if total_cost is not None else "Cost information not available"

        # Salva os resultados
        new_record = {
            "Question": request.question,
            "Original Answer": request.answer,
            "Correct": request.correct,
            "Original Score": previous_score,
            "Original Feedback": request.feedback,
            "Suggestions": suggestions,
            "Revised Answer": revised_answer,
            "Final Score": new_score,
            "Final Answer": final_answer,
            "Decision": decision,
            "Justification": decision_justification,
            "Number of Revisions": number_of_revisions,
            "Language": language,
            "Intent": intent,
            "Category": request.category,
        }
        self.save_result(new_record)

        return {
            "final_answer": final_answer,
            "previous_score": previous_score,
            "new_score": new_score,
        }
    
    def process_revisions(self, requests: List[RevisionRequest]) -> List[str]:
        """
        Processes a list of revision requests.
        Returns a list of final revised answers.
        """
        responses = []
        for req in requests:
            revised = self.process_revision(req)
            responses.append(revised)

        return responses

    def save_result(self, record):
        """
        Reads existing records (if any) and adds a new record,
        saving everything to the CSV file.
        """
        # Check if the file exists and if it is empty
        file_exists = os.path.exists(self.results_file)
        is_empty = not file_exists or os.stat(self.results_file).st_size == 0

        # Open the file in append mode
        with open(self.results_file, "a", newline="", encoding="utf-8") as f:
            # Use the keys of the record as CSV field names
            fieldnames = list(record.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # If the file does not exist or is empty, write the header row
            if is_empty:
                writer.writeheader()

            # Write the new record to the CSV file
            writer.writerow(record)
