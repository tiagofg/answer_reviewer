import os
import re
import autogen
from dotenv import load_dotenv

load_dotenv()

# LLM model configuration
config_list = [
    {
        "model": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
]
llm_config = {"config_list": config_list, "temperature": 0.0}

# Reviewer Agent: evaluates the answer and suggests improvements (does not provide the final answer).
reviewer = autogen.AssistantAgent(
    name="Reviewer",
    llm_config=llm_config,
    max_consecutive_auto_reply=2,
    is_termination_msg=lambda msg: "It is not possible to provide a revised answer." in msg.get("content", ""),
    system_message=(
        "You are an AI assistant whose purpose is to review the quality of an answer provided "
        "for a question asked to the user regarding a product. "
        "The question may have different intentions, the closest match will be provided along with the question and the answer. "
        "The questions and answers may be in Portuguese or Spanish, but your scores and suggestions must be in English. "
        "You must evaluate two main aspects: whether the answer is semantically correct and whether the answer is contextually correct. "
        "Along with the question and the answer, context will be provided that must be taken into account for the evaluation. "
        "As it will also be provided the metadata, that contains some information and rules for the evaluation, you must take it into account. "
        "You must provide a score from 0 to 5 for each aspect, and the final score will be the sum of the two scores. "
        "The semantic score should be available in the message, between the tags <semantic_score> and </semantic_score>. "
        "The contextual score should be available in the message, between the tags <contextual_score> and </contextual_score>. "
        "The final score should be available in the message, between the tags <total_score> and </total_score>. "
        "If the final score is 7 or less, you must present the points that are incorrect and suggest what should be done to improve the answer. "
        "The sugestions must be provided in the message, between the tags <suggestions> and </suggestions>. "
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
    is_termination_msg=lambda msg: bool(
        (m := re.search(r"<total_score>(\d+)</total_score>", msg.get("content", "")))
        and int(m.group(1)) > 7
    ),
    system_message=(
        "You must send a set of questions and answers to be evaluated by an AI assistant. "
        "The question may have different intentions, the closest match will be provided along with the question and the answer. "
        "The questions and answers may be in Portuguese or Spanish; when rewriting the answer, you must consider the original language of the question. "
        "If the final score provided by the reviewer is less than 6, the it will present the points that are incorrect and suggest what should be done to improve the answer. "
        "You must make the suggested corrections and return the corrected answer to be evaluated again. "
        "The revised answer must be provided in the message, between the tags <revised_answer> and </revised_answer>. "
        "If you don't have enough information in the context to answer the question, you need return the following text: THIS QUESTION CANNOT BE ANSWERED!!. "
        "If the answer contained some type of greeting or signature, you must keep it in the revised answer. "
    ),
    code_execution_config={
        "use_docker": False,
    }
)
