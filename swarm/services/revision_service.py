import os
import csv
import json

from typing import List

from autogen import AfterWorkOption, initiate_swarm_chat

from models.revision import RevisionRequest
# Import the necessary agents
from agents.agents import semantic_reviewer, contextual_reviewer, suggester, rewriter, decider, user_proxy

context_variables = dict()  # Variável global para armazenar os context variables

class RevisionService:
    def __init__(self, results_file: str = "results.csv"):
        self.results_file = results_file

    def process_revision(self, request: RevisionRequest) -> str:
        global context_variables  # Referência à variável global

        # Construir os dados da pergunta
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

        # Limpa o dicionário global in-place
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
        })

        # Chama o swarm chat, passando a mesma referência global de context_variables
        chat_history, context_variables, last_active_agent = initiate_swarm_chat(
            user_agent=user_proxy,
            initial_agent=semantic_reviewer,
            agents=[semantic_reviewer, contextual_reviewer, suggester, rewriter, decider],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "When you call the semantic_reviewer for the first time, you must pass the following parameters: "
                        "question, category, language, intent, and original_answer. "
                        "When you call the contextual_reviewer for the first time, you must pass the following parameters: "
                        "question, category, language, context, metadata, and original_answer. "
                        "When you call the suggester for the first time, you must pass the following parameters: "
                        "question, category, context, metadata, semantic_score, justification_semantic, contextual_score, "
                        "justification_contextual, and original_answer. "
                        "When you call the rewriter for the first time, you must pass the following parameters: "
                        "question, category, language, context, metadata, suggestions, and original_answer. "
                        "After the rewriter generates the answer, you must call semantic_reviewer again to evaluate the new answer. "
                        "After the semantic_reviewer has finished, you must call the contextual_reviewer again to evaluate the new answer. "
                        "This is the initial data that you need to pass to the agents: "
                        f"{formatted_question} "
                    )
                }
            ],
            context_variables=context_variables,
            after_work=AfterWorkOption.TERMINATE,
        )

        final_answer = context_variables.get("final_answer")
        previous_score = context_variables.get("original_score")
        new_score = context_variables.get("new_score")
        suggestions = context_variables.get("suggestions")
        revised_answer = context_variables.get("revised_answer")

        if (new_score is not None) and (new_score <= 7):
            final_answer = "DO_NOT_ANSWER"

        if (previous_score is not None) and (previous_score > 7):
            final_answer = request.answer

        new_score = new_score if new_score is not None else "-"
        final_answer = final_answer if final_answer != "DO_NOT_ANSWER" else "-"

        # Salva os resultados
        new_record = {
            "Question": request.question,
            "Original Answer": request.answer,
            "Original Score": previous_score,
            "Original Feedback": request.feedback,
            "Suggestions": suggestions,
            "Revised Answer": revised_answer,
            "Final Score": new_score,
            "Final Answer": final_answer,
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
