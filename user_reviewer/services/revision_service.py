import os
import csv
import json
import re
from typing import List
from models.revision import RevisionRequest
from agents.agents import reviewer, user_proxy  # Import the necessary agents


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
            "metadata": request.metadata,
            "category": request.category,
            "language": language,
            "intent": intent,
        }

        formatted_question = json.dumps(
            question_data, indent=2, ensure_ascii=False)
        message = f"Please evaluate the following answer:\n{formatted_question}"

        # Start the chat for evaluation/revision
        result = user_proxy.initiate_chat(reviewer, message=message)

        # Extract relevant information from the chat history
        final_answer, previous_score, new_score, suggestions = self.extract_chat_results(
            result, request.answer)

        # Extract total cost, if available
        total_cost = result.cost.get(
            'usage_excluding_cached_inference', {}).get('total_cost')
        cost_str = f"${total_cost}" if total_cost is not None else "Cost information not available"

        # Define the 'revised_answer' field based on the rules
        revised_answer = self.determine_revised_answer(
            request.answer, final_answer)

        new_score = new_score if new_score is not None else "-"

        # Build the record to save the results
        new_record = {
            "Question": request.question,
            "Original Answer": request.answer,
            "Original Score": previous_score,
            "Original Feedback": request.feedback,
            "Suggestions": suggestions,
            "Revised Answer": revised_answer,
            "Final Score": new_score,
            "Total Cost": cost_str,
            "Language": language,
            "Intent": request.intent.get("name"),
            "Category": request.category,
        }

        self.save_result(new_record)

        return final_answer.strip()

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
        Extracts final answer, scores, and suggestions from chat using regex.
        Returns: final_answer, previous_score, new_score, suggestions.
        """
        chat = getattr(result, "chat_history", []) or []
        content = "\n".join(msg.get("content", "") for msg in chat if msg.get("name") in ("Reviewer", "User"))

        # Final answer extraction
        revised_match = re.search(r"<revised_answer>(.*?)</revised_answer>", content, re.DOTALL)
        final_answer = revised_match.group(1).strip() if revised_match else original_answer

        # Scores extraction
        prev_match = re.search(r"<total_score>(\d+)</total_score>", content)
        scores = [int(prev_match.group(1))] if prev_match else []
        # If two scores (original and updated) appear, capture both
        all_scores = re.findall(r"<total_score>(\d+)</total_score>", content)
        previous_score = int(all_scores[0]) if len(all_scores) > 0 else None
        new_score = int(all_scores[1]) if len(all_scores) > 1 else None

        # Suggestions extraction
        sug_match = re.search(r"<suggestions>(.*?)</suggestions>", content, re.DOTALL)
        suggestions = sug_match.group(1).strip() if sug_match else None

        return final_answer, previous_score, new_score, suggestions

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
