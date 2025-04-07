import os
import autogen
import re

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
    max_consecutive_auto_reply=1,
    system_message=(
        "You are an AI assistant whose purpose is to review the quality of an answer provided for a question asked to the user regarding a product. "
        "This question may have different intents, the closest match to question will be provided along with the question and the answer. "
        "You will also receive a metadata containing some informations and rules for the answer, that should be taken into account. "
        "Another important information is the category, it describes the category of the product related to the question. "
        "The questions and answers may be in Portuguese or Spanish, but your scores and suggestions must be in English. "
        "You must evaluate two main aspects: whether the answer is semantically correct and whether the answer is contextually correct. "
        "To consider an answer semantically correct, it must explicitly address the question asked and grammatically correct. "
        "To consider an answer contextually correct, it must have the correct information according to the context or metadata provided. "
        "You must provide a score from 0 to 5 for each aspect, and the final score will be the sum of the two scores. "
        "If the answer mentions that there isn't enough information to provide a correct answer, it must not be considered contextually correct. "
        "So a question that has missing or incorrect information should not get a score 4 or 5 for the contextual score. "
        "If the final score is 7 or less, you must present the points that are incorrect and suggest what should be done to improve the answer. "
        "The semantic score should be available in the message, between the tags <semantic_score> and </semantic_score>. "
        "The contextual score should be available in the message, between the tags <contextual_score> and </contextual_score>. "
        "The final score should be available in the message, between the tags <total_score> and </total_score>. "
        "The sugestions must be in provided in the message, between the tags <suggestions> and </suggestions>. "
        "If the final score is higher than 7, you don't need to provide any suggestions. "
        "You must not provide a revised answer, only suggestions for improvement. "
    )
)

rewriter = autogen.AssistantAgent(
    name="Rewriter",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
    system_message=(
        "You are an AI assistant whose purpose is to rewrite answers that have not been evaluated positively by the reviewer. "
        "You will receive the original question, the original answer, and the suggestions for improvement made by the reviewer. "
        "Other important informations that you should use to rewrite the answer are the context, the category, the intent and the metadata. "
        "The context is a object that contains the information about the product, the store and other useful informations. "
        "The category is a string that describes the category of the product related to the question. "
        "The intent is a object that contains the possible intents of the question, calculated based on the question. "
        "The metadata is a object that contains some informations and rules for the answer, that should be taken into account. "
        "The questions and answers may be in Portuguese or Spanish, but your revised answer must be in the original language of the question. "
        "You must consider the suggestions made by the reviewer and rewrite the answer accordingly. "
        "If the answer contained some type of greeting or signature, you must keep it in the revised answer. "
        "If you don't have information in the context to answer the question, you need return the following text: THIS QUESTION CANNOT BE ANSWERED!!. "
        "If there is a clear statement in the context or in the metadata that says that this type of question shouldn't be answered, "
        "you must return the following text: THIS QUESTION CANNOT BE ANSWERED!!. "
        "You must use only information that can be explicitly inferred from the context, and that makes sense for the question asked. "
        "The revised answer should provided in the message, between the tags <revised_answer> and </revised_answer>. "
    ),
)

evaluator = autogen.AssistantAgent(
    name="Evaluator",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
    system_message=(
        "You are an AI assistant whose purpose is to evaluate an answer given for a question asked by a costumer regarding a product. "
        "If the answer given was not evaluated positively by the reviewer, a new answer was written by the rewriter. "
        "Your goal is to evaluate if the rewritten answer is an improvement over the original answer. "
        "If you consider that none of the answers directly address the question, you must return the following text: THIS QUESTION CANNOT BE ANSWERED!!. "
        "Followed by the text, None of the answers are good enough to be accepted. "
        "You must not accept an answer that mentions that there isn't information available to answer the user's question. "
        "You must not accept an answer that mentions another product, unless it is mentioned in the context or metadata, containing a link to the product. "
        "You must not accept an answer that says that the anser cannot be answered. "
        "If any of these situations occur, you must return the following text: THIS QUESTION CANNOT BE ANSWERED!!. "
        "Followed by the reason why the answer cannot be accepted. "
        "If you consider that the rewritten answer is an improvement over the original answer, you must return the new answer. "
        "If you consider that the rewritten answer is not an improvement over the original answer, you must return the original answer. "
        "You should also provide a score from 0 to 10 for the chosen answer."
        "The score should be provided in the message, between the tags <new_score> and </new_score>. "
        "The answer should be provided in the message, between the tags <final_answer> and </final_answer>. "
        "If the score is 5 or less, you must only return the text: THIS QUESTION CANNOT BE ANSWERED!!. "
        "Followed by the text, 'The revised answer is not good enough to be accepted' and the score you gave it"
    )
)

# User Agent: sends the question for evaluation and, if necessary, revises the answer according to the reviewer's suggestions.
# The final answer (original or revised) must be provided by user_proxy.
user_proxy = autogen.UserProxyAgent(
    name="User",
    llm_config=llm_config,
    human_input_mode="NEVER",
    system_message=(
        "You must send a answer given for a question asked by a costumer regarding a product for evaluation. "
        "The object that you will send contains the question, the answer, the context, the category, the metadata, the language and the intent. "
        "This question needs to be evaluated by the reviewer, and if necessary, revised by the rewriter. "
        "The revised answer must be evaluated by the evaluator. "
    ),
    code_execution_config={
        "use_docker": False,
    }
)

group_chat = autogen.GroupChat(
    agents=[reviewer, rewriter, evaluator],
    speaker_selection_method="round_robin",
)

manager = autogen.GroupChatManager(
    groupchat=group_chat,
    is_termination_msg=lambda x: (
        (x.get("content", "").find("THIS QUESTION CANNOT BE ANSWERED!!") >= 0) or
        (lambda m: int(m.group(1)) > 7 if m else False)(re.search(r"<total_score>(\d+)</total_score>", x.get("content", "")))
    ),
    llm_config=llm_config,
    system_message=(
        "You are the manager of a group chat that contains three AI assistants: the reviewer, the rewriter, and the evaluator. "
        "The reviewer evaluates the quality of an answer provided for a question asked by the user regarding a product. "
        "The rewriter rewrites answers that have not been evaluated positively by the reviewer. "
        "The evaluator evaluates if the rewritten answer is an improvement over the original answer. "
        "You must manage the conversation between the assistants and make sure that the final answer is provided to the user. "
        "Each assistant must speak only once in the conversation, and they must speak in the following order: reviewer, rewriter, evaluator. "
    )
)
