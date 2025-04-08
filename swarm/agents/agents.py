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
    }
]
llm_config = {"config_list": config_list}

def register_semantic_score(semantic_score: int, justification: str, context_variables: dict) -> autogen.SwarmResult:
    """
    Register the semantic score and justification in the context variables.
    """
    context_variables["semantic_score"] = semantic_score
    context_variables["justification_semantic"] = justification

    return autogen.SwarmResult(
        context_variables=context_variables,
        agent=contextual_reviewer,
    )

def register_contextual_score(contextual_score: int, justification: str, context_variables: dict) -> autogen.SwarmResult:
    """
    Register the contextual score and justification in the context variables.
    """
    context_variables["contextual_score"] = contextual_score
    context_variables["justification_contextual"] = justification

    score = (contextual_score + context_variables["semantic_score"]) // 2

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
    context_variables["revised_answer"] = revised_answer

    return autogen.SwarmResult(
        context_variables=context_variables,
        agent=semantic_reviewer,
        message=(
            f"Please evaluate this answer now\n{revised_answer}"
        ),
    )

semantic_reviewer = autogen.AssistantAgent(
    name="Semantic_Reviewer",
    llm_config=llm_config,
    system_message=(
        "You are an AI assistant whose purpose is to review the semantic of an answer provided for a question asked to the user regarding a product. "
        "This question may have different intents, the closest match to question will be provided along with the question and the answer. "
        "You will also receive a metadata containing some informations and rules for the answer, that should be taken into account. "
        "Another important information is the category, it describes the category of the product related to the question. "
        "You must evaluate whether the answer is semantically correct. "
        "To consider an answer semantically correct, it must explicitly address the question asked and be grammatically correct. "
        "You must provide a score from 0 to 10 for the semantic aspect, together with a brief justification. "
        "Your response must be in English and have the following format: "
        "<semantic_score>{score}</semantic_score>\n<justification>{justification}</justification>. "
    ),
    functions=[register_semantic_score],
)

contextual_reviewer = autogen.AssistantAgent(
    name="Contextual_Reviewer",
    llm_config=llm_config,
    system_message=(
        "You are an AI assistant whose purpose is to review the contextual of an answer provided for a question asked to the user regarding a product. "
        "This question may have different intents, the closest match to question will be provided along with the question and the answer. "
        "You will also receive a metadata containing some informations and rules for the answer, that should be taken into account. "
        "Another important information is the category, it describes the category of the product related to the question. "
        "You must evaluate whether the answer is contextually correct. "
        "To consider an answer contextually correct, it must have the correct information according to the context or metadata provided. "
        "You must provide a score from 0 to 10 for the contextual aspect, together with a brief justification. "
        "Your response must be in English and have the following format: "
        "<contextual_score>{score}</contextual_score>\n<justification>{justification}</justification>. "
    ),
    functions=[register_contextual_score],
)

suggester = autogen.AssistantAgent(
    name="Suggester",
    llm_config=llm_config,
    system_message=(
        "You are an AI assistant whose purpose is to suggest improvements for an answer provided for a question asked to the user regarding a product. "
        "You must provide suggestions for improvement based on the semantic and contextual scores provided by the reviewers. "
        "They will also provide a brief justification for the scores, you can use this information to provide better suggestions. "
        "You must not provide a revised answer, only suggestions for improvement. "
        "Your suggestions must be in English and have the following format: <suggestions>{suggestions}</suggestions>. "
    ),
    functions=[register_suggestions],
)

rewriter = autogen.AssistantAgent(
    name="Rewriter",
    llm_config=llm_config,
    system_message=(
        "You are an AI assistant whose purpose is to rewrite answers that have not been evaluated positively by the reviewers. "
        "You will receive the original question, the original answer, and the suggestions for improvement made by the suggester. "
        "Other important informations that you should use to rewrite the answer are the context, the category, the intent and the metadata. "
        "The context is a object that contains the information about the product, the store and other useful informations. "
        "The category is a string that describes the category of the product related to the question. "
        "The intent is a object that contains the possible intents of the question, calculated based on the question. "
        "The metadata is a object that contains some informations and rules for the answer, that should be taken into account. "
        "The questions and answers may be in Portuguese or Spanish, but your revised answer must be in the original language of the question. "
        "If the answer contained some type of greeting or signature, you must keep it in the revised answer. "
        "Your revised answer must have the following format: "
        "<revised_answer>{revised_answer}</revised_answer>. "
    ),
    functions=[register_revised_answer],
)

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


