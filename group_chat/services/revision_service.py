import os
import csv
import json
from typing import List

import autogen
from models.revision import RevisionRequest
from agents.agents import manager, user_proxy  # Import the necessary agents


class RevisionService:
    def __init__(self, results_file: str = "results.csv"):
        self.results_file = results_file

    def process_revision(self, request: RevisionRequest) -> str:
        """
        Processes a single revision request.
        Returns the final revised answer.
        """
        # Extract the language and intent from the request
        language = "portuguese" if request.locale == "pt" else "spanish"
        intent = request.intent.get("name")

        # Build the JSON with the question fields
        question_data = {
            "question": request.question,
            "answer": request.answer,
            "context": request.context,
            "category": request.category,
            "metadata": request.metadata,
            "language": language,
            "intent": intent,
        }

        formatted_question = json.dumps(
            question_data, indent=2, ensure_ascii=False)
        message = f"Please send this answer to be reviewed\n{formatted_question}"

        result = user_proxy.initiate_chat(recipient=manager, message=message)

        # # Extract relevant information from the chat history
        # final_answer, score, previous_score, new_score, suggestions = self.extract_chat_results(
        #     result, request.answer)

        # # Extract total cost, if available
        # total_cost = result.cost.get(
        #     'usage_excluding_cached_inference', {}).get('total_cost')
        # cost_str = f"${total_cost}" if total_cost is not None else "Cost information not available"

        # # Define the 'revised_answer' field based on the rules
        # revised_answer = self.determine_revised_answer(
        #     request.answer, final_answer)

        # new_score = new_score if new_score is not None else "-"
        # previous_score = previous_score if previous_score is not None else score

        # # Build the record to save the results
        # new_record = {
        #     "Question": request.question,
        #     "Original Answer": request.answer,
        #     "Original Score": previous_score,
        #     "Original Feedback": request.feedback,
        #     "Suggestions": suggestions,
        #     "Revised Answer": revised_answer,
        #     "Final Score": new_score,
        #     "Total Cost": cost_str,
        #     "Language": language,
        #     "Intent": request.intent.get("name"),
        #     "Category": request.category,
        # }

        # self.save_result(new_record)

        # return final_answer.strip()

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

    @staticmethod
    def extract_chat_results(result, original_answer):
        """
        Extracts relevant information from the chat history.
        Returns: final_answer, score, previous_score, new_score, suggestions.
        """
        final_answer = original_answer
        score = None
        previous_score = None
        new_score = None
        suggestions = None

        if hasattr(result, "chat_history") and isinstance(result.chat_history, list):
            for msg in reversed(result.chat_history):
                content = msg.get("content", "")

                # Process the revised answer sent by "User"
                if msg.get("role") == "assistant" and msg.get("name") == "User":
                    if "Revised Answer:" in content and final_answer == original_answer:
                        final_answer = content.split("Revised Answer:")[
                            1].strip().replace('"', '')
                    elif "It is not possible to provide a revised answer." in content:
                        final_answer = "It is not possible to provide a revised answer."

                # Process the information from the "Reviewer"
                elif msg.get("name") == "Reviewer":
                    if "Final Score" in content:
                        score_value = int(content.split(
                            "Final Score: ")[1].split("/")[0])

                        if score is None:
                            score = score_value
                        else:
                            new_score = score
                            previous_score = score_value

                    if "Suggestions:" in content and suggestions is None:
                        suggestions = content.split("Suggestions:")[1].strip()

        return final_answer, score, previous_score, new_score, suggestions

    @staticmethod
    def determine_revised_answer(original, revised):
        """
        Defines the 'Revised Answer' field based on the logic:
          - If the revised answer is equal to the original, return "-"
          - If it is not possible to revise, return None (null in JSON)
          - Otherwise, return the revised answer.
        """
        if revised == original:
            return "-"
        elif revised == "It is not possible to provide a revised answer.":
            return None

        return revised

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
