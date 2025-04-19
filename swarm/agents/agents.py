import os
from typing import Any, Dict, List, Union
import autogen
import re

from dotenv import load_dotenv

load_dotenv()

# LLM model configuration
config_list = [
    {
        "model": "gpt-4o",
        "api_key": os.getenv("OPEN_AI_API_KEY"),
    }
]
llm_config = {"config_list": config_list, "temperature": 0.0}

def register_semantic_score(semantic_score: int, justification: str, context_variables: dict) -> autogen.SwarmResult:
    """
    Register the semantic score and justification in the context variables.
    """
    if (context_variables.get("revised_answer") is None):
        context_variables["semantic_score"] = semantic_score
        context_variables["justification_semantic"] = justification
    else:
        context_variables["revised_answer_semantic_score"] = semantic_score
        context_variables["revised_answer_justification_semantic"] = justification

    return autogen.SwarmResult(
        context_variables=context_variables,
        agent=contextual_reviewer,
    )

def register_contextual_score(contextual_score: int, justification: str, context_variables: dict) -> autogen.SwarmResult:
    """
    Register the contextual score and justification in the context variables.
    """
    original_score = None
    new_score = None

    if (context_variables.get("revised_answer") is None):
        context_variables["contextual_score"] = contextual_score
        context_variables["justification_contextual"] = justification

        semantic_score = context_variables["semantic_score"]
        original_score = (contextual_score + semantic_score) / 2

        context_variables["original_score"] = original_score
    else:
        context_variables["revised_answer_contextual_score"] = contextual_score
        context_variables["revised_answer_justification_contextual"] = justification

        semantic_score = context_variables["revised_answer_semantic_score"]
        new_score = (contextual_score + semantic_score) / 2

        context_variables["new_score"] = new_score

    if original_score is not None and original_score > 7.0:
        return autogen.SwarmResult(
            context_variables=context_variables,
            after_work=autogen.AfterWork(autogen.AfterWorkOption.TERMINATE),
        )
    elif new_score is not None:
        return autogen.SwarmResult(
            context_variables=context_variables,
            agent=decider,
        )
    else:
        return autogen.SwarmResult(
            context_variables=context_variables,
            agent=suggester,
        )

def register_suggestions(suggestions: str, context_variables: dict) -> autogen.SwarmResult:
    """
    Register the suggestions in the context variables.
    """
    context_variables["suggestions"] = suggestions

    return autogen.SwarmResult(
        context_variables=context_variables,
        agent=rewriter,
    )

def register_revised_answer(revised_answer: str, context_variables: dict) -> autogen.SwarmResult:
    """
    Register the revised answer in the context variables.
    """
    context_variables["revised_answer"] = revised_answer

    if re.search(r"^CANNOT REWRITE$", revised_answer):
        context_variables["final_answer"] = "DO_NOT_ANSWER"

        return autogen.SwarmResult(
            context_variables=context_variables,
        )

    return autogen.SwarmResult(
        context_variables=context_variables,
        agent=semantic_reviewer,
    )

def register_decision(decision: str, context_variables: dict) -> autogen.SwarmResult:
    """
    Register the decision in the context variables.
    """
    context_variables["decision"] = decision

    if decision == "ANSWER_ORIGINAL":
        context_variables["final_answer"] = context_variables["original_answer"]

        return autogen.SwarmResult(
            context_variables=context_variables,
        )
    elif decision == "ANSWER_REVISED":
        context_variables["final_answer"] = context_variables["revised_answer"]

        return autogen.SwarmResult(
            context_variables=context_variables,
        )
    elif decision == "REWRITE":
        context_variables["answer"] = context_variables["revised_answer"]
        context_variables["semantic_score"] = context_variables["revised_answer_semantic_score"]
        context_variables["justification_semantic"] = context_variables["revised_answer_justification_semantic"]
        context_variables["contextual_score"] = context_variables["revised_answer_contextual_score"]
        context_variables["justification_contextual"] = context_variables["revised_answer_justification_contextual"]
        context_variables["revised_answer"] = None
        context_variables["revised_answer_semantic_score"] = None
        context_variables["revised_answer_justification_semantic"] = None
        context_variables["revised_answer_contextual_score"] = None
        context_variables["revised_answer_justification_contextual"] = None
        context_variables["suggestions"] = None
        context_variables["new_score"] = None

        return autogen.SwarmResult(
            context_variables=context_variables,
            agent=rewriter,
        )
    
    context_variables["final_answer"] = "DO_NOT_ANSWER"

    return autogen.SwarmResult(
        context_variables=context_variables,
    )

semantic_reviewer = autogen.AssistantAgent(
    name="Semantic_Reviewer",
    llm_config=llm_config,
    system_message=(
        "You are the Semantic Reviewer whose purpose is to review the semantic of an answer provided for a question asked to the user regarding a product. "
        "Or a revised answer that was written by the rewriter to try to improve the original answer. "
        "The question may have different intents, the closest match to question will be provided along with the question and the answer. "
        "Another important information is the category, it describes the category of the product related to the question. "
        "All the data that you need, can be found in the context_variables. "
        "You must evaluate only whether the answer is semantically correct, not anything related to the context. "
        "To consider an answer semantically correct, it must explicitly address the question asked and be grammatically correct. "
        "You must provide a score from 0 to 10 for the semantic aspect, together with a brief justification in English. "
        "You must always call the function register_semantic_score, passing the semantic score and justification as parameters. "
    ),
    functions=[register_semantic_score],
    max_consecutive_auto_reply=6,
)

contextual_reviewer = autogen.AssistantAgent(
    name="Contextual_Reviewer",
    llm_config=llm_config,
    system_message=(
        "You are the Contextual Reviewer whose purpose is to review the contextual of an answer provided for a question asked to the user regarding a product. "
        "Or a revised answer that was written by the rewriter to try to improve the original answer. "
        "You will also receive a metadata containing some informations and rules for the answer, that should be taken into account. "
        "Another important information is the category, it describes the category of the product related to the question. "
        "As is the context, as it contains the information about the product, the store and other useful informations. "
        "All the data that you need, can be found in the context_variables. "
        "You must evaluate only whether the answer is contextually correct, not anything related to the semantic. "
        "To consider an answer contextually correct, it must have the correct information according to the context or metadata provided. "
        "You must provide a score from 0 to 10 for the contextual aspect, together with a brief justification in English. "
        "You must always call the function register_contextual_score, passing the contextual score and justification as parameters. "
    ),
    functions=[register_contextual_score],
    max_consecutive_auto_reply=6,
)

suggester = autogen.AssistantAgent(
    name="Suggester",
    llm_config=llm_config,
    system_message=(
        "You are the Suggester whose purpose is to suggest improvements for an answer provided for a question asked to the user regarding a product. "
        "You must provide suggestions for improvement based on the semantic and contextual scores provided by the reviewers. "
        "They will also provide a brief justification for the scores, you can use this information to provide better suggestions. "
        "All the data that you need, can be found in the context_variables. "
        "You must not provide a revised answer, only suggestions for improvement. "
        "The suggestions must be in English, even if the question and answer are in Portuguese or Spanish. "
        "You must always call the function register_suggestions, passing your suggestions as parameter. "
    ),
    functions=[register_suggestions],
    max_consecutive_auto_reply=3,
)

rewriter = autogen.AssistantAgent(
    name="Rewriter",
    llm_config=llm_config,
    system_message=(
        "You are the Rewriter whose purpose is to rewrite answers that have not been evaluated positively by the reviewers. "
        "You will receive the original question, the original answer, and the suggestions for improvement made by the suggester. "
        "Other important informations that you should use to rewrite the answer are the context, the category, the intent and the metadata. "
        "The context is a object that contains the information about the product, the store and other useful informations. "
        "The category is a string that describes the category of the product related to the question. "
        "The intent is a object that contains the possible intents of the question, calculated based on the question. "
        "The metadata is a object that contains some informations and rules for the answer, that should be taken into account. "
        "All the data that you need, can be found in the context_variables. "
        "The questions and answers may be in Portuguese or Spanish, your revised answer must be in the original language of the question. "
        "If the answer contained some type of greeting or signature, you must keep it in the revised answer. "
        "If you consider that there isn't enough information to provide a revised answer, you must return the text 'CANNOT REWRITE'. "
        "You must always call the function register_revised_answer, passing the revised answer as parameter. "
    ),
    functions=[register_revised_answer],
    max_consecutive_auto_reply=3,
)

decider = autogen.AssistantAgent(
    name="Decider",
    llm_config=llm_config,
    system_message=(
        "You are the Decider whose purpose is to decide whether the answer provided for the user is good enough or can be improved. "
        "If the reviewers evaluated the answer positively, you will receive the original answer and the scores. "
        "If the reviewers evaluated the answer negatively, you will receive the revised answer and the scores. "
        "You must decide whether the answer is good enough to be answered or not. "
        "You must always call the function register_decision, passing your decision as parameter. "
        "If the answer is not good enough and based in the information you think that the rewriter can improve it, your decision must be 'REWRITE'. "
        "If the answer is not good enough and based in the information you think that is not possible to improve, your decision must be 'DO_NOT_ANSWER'. "
        "If the answer says that there is no information to fully address the question, your decision must be 'DO_NOT_ANSWER'. "
        "If the original answer is good enough, your decision must be 'ANSWER_ORIGINAL'. "
        "If the revised answer is good enough, your decision must be 'ANSWER_REVISED'. "
    ),
    functions=[register_decision],
    max_consecutive_auto_reply=5,
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config={
        "use_docker": False,
    }
)

autogen.register_hand_off(
    agent=rewriter,
    hand_to=[autogen.AfterWork(autogen.AfterWorkOption.TERMINATE)]
)

autogen.register_hand_off(
    agent=decider,
    hand_to=[autogen.AfterWork(autogen.AfterWorkOption.TERMINATE)]
)
