import os
import autogen
from dotenv import load_dotenv

load_dotenv()

# LLM model configuration
config_list = [
    {
        "model": "gpt-4o",
        "api_key": os.getenv("OPEN_AI_API_KEY"),
        "temperature": 0,
    }
]
llm_config = {"config_list": config_list}

# Reviewer Agent: evaluates the answer and suggests improvements (does not provide the final answer).
reviewer = autogen.AssistantAgent(
    name="Reviewer",
    llm_config=llm_config,
    max_consecutive_auto_reply=2,
    is_termination_msg=lambda msg: "It is not possible to provide a revised answer." in msg.get("content", ""),
    system_message=(
        "You are an AI assistant whose purpose is to review the quality of an answer provided "
        "for a question asked to the user regarding a product. This question can have 3 different intentions: "
        "1. Compatibility: the user asks if the product is compatible with another product. "
        "2. Specification: the user asks about the product's specifications. "
        "3. Availability: the user asks about the product's availability. "
        "The questions and answers may be in Portuguese or Spanish; when reviewing the answer, you must consider the original language of the question. "
        "You must evaluate two main aspects: whether the answer is semantically correct and whether the answer is contextually correct. "
        "Along with the question and the answer, context will be provided that must be taken into account for the evaluation. "
        "You must provide a score from 0 to 5 for each aspect, and the final score will be the sum of the two scores. "
        "For each score, you must put the number of the score followed by a slash and the total possible score. "
        "If the final score is 7 or less, you must present the points that are incorrect and suggest what should be done to improve the answer. "
        "The sugestions must be in the end of the message, after the text 'Suggestions:'. "
        "If the final score is higher than 7, you don't need to provide any suggestions. "
        "You must not provide a revised answer, the user will make the necessary corrections and return the corrected answer for evaluation. "
    )
)

# User Agent: sends the question for evaluation and, if necessary, revises the answer according to the reviewer's suggestions.
# The final answer (original or revised) must be provided by user_proxy.
user_proxy = autogen.UserProxyAgent(
    name="User",
    llm_config=llm_config,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    is_termination_msg=lambda msg: "Final Score" in msg.get("content", "") and int(
        msg.get("content", "").split("Final Score: ")[1].split("/")[0]
    ) > 7,
    system_message=(
        "You must send a set of questions and answers to be evaluated by an AI assistant. "
        "Each question can have 3 different intentions: "
        "1. Compatibility: the user asks if the product is compatible with another product. "
        "2. Specification: the user asks about the product's specifications. "
        "3. Availability: the user asks about the product's availability. "
        "The questions and answers may be in Portuguese or Spanish; when rewriting the answer, you must consider the original language of the question. "
        "If the final score provided by the reviewer is less than 6, the it will present the points that are incorrect and suggest what should be done to improve the answer. "
        "You must make the suggested corrections and return the corrected answer to be evaluated again. "
        "Immediately after the text 'Revised Answer:', you must provide the corrected answer. This should be the end of the message. "
        "If you believe that, given the context, it is not possible to provide a correct answer, you must return the following text: "
        "'It is not possible to provide a revised answer.' "
        "If the answer contained some type of greeting or signature, you must keep it in the revised answer. "
    ),
    code_execution_config={
        "use_docker": False,
    }
)
