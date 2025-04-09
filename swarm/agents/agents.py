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
llm_config = {"config_list": config_list}

def register_semantic_score(semantic_score: int, justification: str, context_variables: dict) -> autogen.SwarmResult:
    """
    Register the semantic score and justification in the context variables.
    """
    if (not context_variables.get("revised_answer")):
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
    if (not context_variables.get("revised_answer")):
        context_variables["contextual_score"] = contextual_score
        context_variables["justification_contextual"] = justification
        new_contextual_score = contextual_score
        new_semantic_score = context_variables["semantic_score"]
    else:
        context_variables["revised_answer_contextual_score"] = contextual_score
        context_variables["revised_answer_justification_contextual"] = justification
        new_contextual_score = contextual_score
        new_semantic_score = context_variables["revised_answer_semantic_score"]

    score = (new_contextual_score + new_semantic_score) / 2

    if score > 7:
        return autogen.SwarmResult(
            context_variables=context_variables,
            after_work=autogen.AfterWorkOption.TERMINATE,
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
    context_variables["answer"] = revised_answer
    context_variables["revised_answer"] = revised_answer

    return autogen.SwarmResult(
        context_variables=context_variables,
        agent=semantic_reviewer,
        message=(
            "Please ask the reviewers to evaluate the revised answer. "
            "Using the same criteria as before, but now with the revised answer. "
        ),
    )

semantic_reviewer = autogen.AssistantAgent(
    name="Semantic_Reviewer",
    llm_config=llm_config,
    system_message=(
        "You are the Semantic Reviewer whose purpose is to review the semantic of an answer provided for a question asked to the user regarding a product. "
        "The question may have different intents, the closest match to question will be provided along with the question and the answer. "
        "You will also receive a metadata containing some informations and rules for the answer, that should be taken into account. "
        "Another important information is the category, it describes the category of the product related to the question. "
        "You must evaluate only whether the answer is semantically correct. "
        "To consider an answer semantically correct, it must explicitly address the question asked and be grammatically correct. "
        "You must provide a score from 0 to 10 for the semantic aspect, together with a brief justification. "
        "You may also receive a revised answer, in this case you must evaluate the revised answer using the same criteria. "
        "And also provide a semantic score and justification for it. "
        "You must call the function register_semantic_score, passing the semantic score and justification as parameters. "
    ),
    functions=[register_semantic_score],
)

contextual_reviewer = autogen.AssistantAgent(
    name="Contextual_Reviewer",
    llm_config=llm_config,
    system_message=(
        "You are the Contextual Reviewer whose purpose is to review the contextual of an answer provided for a question asked to the user regarding a product. "
        "The question may have different intents, the closest match to question will be provided along with the question and the answer. "
        "You will also receive a metadata containing some informations and rules for the answer, that should be taken into account. "
        "Another important information is the category, it describes the category of the product related to the question. "
        "As is the context, as it contains the information about the product, the store and other useful informations. "
        "You must evaluate only whether the answer is contextually correct. "
        "To consider an answer contextually correct, it must have the correct information according to the context or metadata provided. "
        "You must provide a score from 0 to 10 for the contextual aspect, together with a brief justification. "
        "You may also receive a revised answer, in this case you must evaluate the revised answer using the same criteria. "
        "And also provide a contextual score and justification for it. "
        "You must call the function register_contextual_score, passing the contextual score and justification as parameters. "
    ),
    functions=[register_contextual_score],
)

suggester = autogen.AssistantAgent(
    name="Suggester",
    llm_config=llm_config,
    system_message=(
        "You are the Suggester whose purpose is to suggest improvements for an answer provided for a question asked to the user regarding a product. "
        "You must provide suggestions for improvement based on the semantic and contextual scores provided by the reviewers. "
        "They will also provide a brief justification for the scores, you can use this information to provide better suggestions. "
        "You must not provide a revised answer, only suggestions for improvement. "
        "The suggestions must be in English, even if the question and answer are in Portuguese or Spanish. "
        "You must call the function register_suggestions, passing the suggestions as parameter. "
    ),
    functions=[register_suggestions],
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
        "The questions and answers may be in Portuguese or Spanish, your revised answer must be in the original language of the question. "
        "If the answer contained some type of greeting or signature, you must keep it in the revised answer. "
        "You must call the function register_revised_answer, passing the revised answer as parameter. "
    ),
    functions=[register_revised_answer],
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config={
        "use_docker": False,
    }
)
