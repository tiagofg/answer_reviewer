import os
import csv
import json
import re
from typing import List

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

        # Extract total cost, if available
        total_cost = result.cost.get(
            'usage_excluding_cached_inference', {}).get('total_cost')
        cost_str = f"${total_cost}" if total_cost is not None else "Cost information not available"

        # Extract relevant information from the chat history
        final_answer, revised_answer, previous_score, new_score, suggestions = self.extract_chat_results(
            manager.chat_messages, request.answer)

        if (new_score is not None) and (new_score <= 7):
            final_answer = "DO_NOT_ANSWER"

        if (previous_score is not None) and (previous_score > 7):
            final_answer = request.answer

        new_score = new_score if new_score is not None else "-"

        final_answer = final_answer if final_answer != "DO_NOT_ANSWER" else "-"

        # Build the record to save the results
        new_record = {
            "Question": request.question,
            "Original Answer": request.answer,
            # "Correct": request.correct,
            "Original Score": previous_score,
            "Original Feedback": request.feedback,
            "Suggestions": suggestions,
            "Revised Answer": revised_answer,
            "Final Score": new_score,
            "Final Answer": final_answer,
            "Language": language,
            "Intent": request.intent.get("name"),
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

    @staticmethod
    def extract_chat_results(messages, original_answer):
        """
        Extrai informações relevantes do histórico do chat.
        Retorna: final_answer, previous_score, new_score, suggestions.
        """
        final_answer = original_answer
        revised_answer = None
        previous_score = None
        new_score = None
        suggestions = None

        # Se 'messages' for um dicionário (por exemplo, defaultdict) com listas de mensagens,
        # "achata" a estrutura em uma única lista
        if isinstance(messages, dict):
            flat_messages = []

            for msg_list in messages.values():
                flat_messages.extend(msg_list)

            messages = flat_messages

        index = 0
        for msg in reversed(messages):
            if index >= 3:
                break

            # Se 'msg' for um dicionário, extrai o conteúdo; caso contrário, usa 'msg' como string
            content = msg.get("content", "") if isinstance(
                msg, dict) else str(msg)

            # Expressões regulares para extrair os conteúdos desejados
            final_answer_match = re.search(
                r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)

            revised_answer_match = re.search(
                r"<revised_answer>(.*?)</revised_answer>", content, re.DOTALL)

            previous_score_match = re.search(
                r"<total_score>(.*?)</total_score>", content)

            new_score_match = re.search(
                r"<new_score>(.*?)</new_score>", content)

            suggestions_match = re.search(
                r"<suggestions>(.*?)</suggestions>", content, re.DOTALL)

            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()

            if revised_answer_match:
                revised_answer = revised_answer_match.group(1).strip()

            if previous_score_match:
                try:
                    previous_score = int(previous_score_match.group(1).strip())
                except ValueError:
                    previous_score = None

            if new_score_match:
                try:
                    new_score = int(new_score_match.group(1).strip())
                except ValueError:
                    new_score = None

            if suggestions_match:
                suggestions = suggestions_match.group(1).strip()

            if content.find("THIS QUESTION CANNOT BE ANSWERED!!") >= 0:
                final_answer = "DO_NOT_ANSWER"

            if (previous_score is not None and new_score is not None) or (previous_score is not None and previous_score > 7):
                break

            index += 1

        return final_answer, revised_answer, previous_score, new_score, suggestions

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
