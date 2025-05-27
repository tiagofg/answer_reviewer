import os
import re

from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv
from autogen.agentchat.group import AgentTarget, ContextVariables, ReplyResult, TerminateTarget

load_dotenv()

config_list = [
    {
        "model": "qwen3:8b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
]
llm_config = {"config_list": config_list, "temperature": 0.0}


def register_semantic_score(semantic_score: int, justification: str, context_variables: ContextVariables) -> ReplyResult:
    """
    Register the semantic score and justification in the context variables.
    """
    if (context_variables.get("revised_answer") is None):
        context_variables["semantic_score"] = semantic_score
        context_variables["justification_semantic"] = justification
    else:
        context_variables["revised_answer_semantic_score"] = semantic_score
        context_variables["revised_answer_justification_semantic"] = justification

    return ReplyResult(
        context_variables=context_variables,
        target=AgentTarget(contextual_reviewer),
        message="The semantic score and justification have been registered, handing over to the Contextual Reviewer for his review.",
    )


def register_contextual_score(contextual_score: int, justification: str, context_variables: ContextVariables) -> ReplyResult:
    """
    Register the contextual score and justification in the context variables.
    """
    original_score = None
    new_score = None

    if (context_variables.get("revised_answer") is None):
        context_variables["contextual_score"] = contextual_score
        context_variables["justification_contextual"] = justification

        semantic_score = context_variables["semantic_score"]
        original_score = (contextual_score + semantic_score)

        context_variables["original_score"] = original_score
    else:
        context_variables["revised_answer_contextual_score"] = contextual_score
        context_variables["revised_answer_justification_contextual"] = justification

        semantic_score = context_variables["revised_answer_semantic_score"]
        new_score = (contextual_score + semantic_score)

        context_variables["new_score"] = new_score

    if original_score is not None and original_score > 8:
        return ReplyResult(
            context_variables=context_variables,
            target=TerminateTarget(),
            message="The original score is greater than 8, terminating the process.",
        )
    elif new_score is not None:
        return ReplyResult(
            context_variables=context_variables,
            target=AgentTarget(decider),
            message="The new score has been registered, handing over to the Decider to make a decision about the revised answer.",
        )
    else:
        return ReplyResult(
            context_variables=context_variables,
            target=AgentTarget(suggester),
            message="The contextual score and justification have been registered, handing over to the Suggester to suggest improvements.",
        )


def register_suggestions(suggestions: str, context_variables: ContextVariables) -> ReplyResult:
    """
    Register the suggestions in the context variables.
    """
    context_variables["suggestions"] = suggestions

    return ReplyResult(
        context_variables=context_variables,
        target=AgentTarget(rewriter),
        message="The suggestions have been registered, handing over to the Rewriter to write a new answer.",
    )


def register_revised_answer(revised_answer: str, context_variables: ContextVariables) -> ReplyResult:
    """
    Register the revised answer in the context variables.
    """
    context_variables["revised_answer"] = revised_answer
    context_variables["number_of_revisions"] += 1

    if re.search(r"^CANNOT REWRITE$", revised_answer):
        context_variables["final_answer"] = "DO_NOT_ANSWER"

        return ReplyResult(
            context_variables=context_variables,
            target=TerminateTarget(),
            message="It's not possible to write a new answer, terminating the process.",
        )

    return ReplyResult(
        context_variables=context_variables,
        target=AgentTarget(semantic_reviewer),
        message="The revised answer has been registered, handing over to the Semantic Reviewer to review it.",
    )


def register_decision(decision: str, justification: str, context_variables: ContextVariables) -> ReplyResult:
    """
    Register the decision in the context variables.
    """
    context_variables["decision"] = decision
    context_variables["decision_justification"] = justification

    if decision == "ANSWER_REVISED":
        context_variables["final_answer"] = context_variables["revised_answer"]

        return ReplyResult(
            context_variables=context_variables,
            target=TerminateTarget(),
            message="The decision is 'ANSWER_REVISED', terminating the process.",
        )
    elif decision == "REWRITE":
        context_variables["original_answer"] = context_variables["revised_answer"]
        context_variables["original_answer_semantic_score"] = context_variables["revised_answer_semantic_score"]
        context_variables["original_answer_justification_semantic"] = context_variables["revised_answer_justification_semantic"]
        context_variables["original_answer_contextual_score"] = context_variables["revised_answer_contextual_score"]
        context_variables["original_answer_justification_contextual"] = context_variables["revised_answer_justification_contextual"]
        context_variables["original_score"] = context_variables["new_score"]
        context_variables["revised_answer"] = None
        context_variables["revised_answer_semantic_score"] = None
        context_variables["revised_answer_justification_semantic"] = None
        context_variables["revised_answer_contextual_score"] = None
        context_variables["revised_answer_justification_contextual"] = None
        context_variables["new_score"] = None

        return ReplyResult(
            context_variables=context_variables,
            target=AgentTarget(rewriter),
            message="The decision is 'REWRITE', handing over to the Rewriter to write a new answer.",
        )

    context_variables["final_answer"] = "DO_NOT_ANSWER"

    return ReplyResult(
        context_variables=context_variables,
        target=TerminateTarget(),
        message="The decision is 'DO_NOT_ANSWER', terminating the process.",
    )


semantic_reviewer = AssistantAgent(
    name="Semantic_Reviewer",
    llm_config=llm_config,
    system_message=(
        "You are the Semantic Reviewer. Your task is to critically evaluate the semantic accuracy of an answer provided to a user's question about a product.\n\n"
        "You will be provided with the following information:\n"
        "- **Question**: The user's inquiry regarding the product.\n"
        "- **Original Answer**: The initial response given to the user's question.\n"
        "- **Revised Answer**: The improved response provided by the Rewriter, if available.\n"
        "- **Category**: The category to which the product belongs.\n"
        "- **Intent**: The identified intent behind the user's question.\n\n"
        "Evaluation Instructions:\n"
        "- If the Revised Answer and the Original Answer are not none, evaluate the Revised Answer and register the score and the justification for it.\n"
        "- If the Revised Answer is none, evaluate the Original Answer and register the score and the justification for it.\n\n"
        "Evaluation Criteria:\n"
        "- The answer must directly and explicitly address all aspects of the user's question.\n"
        "- It must be grammatically correct, free of spelling errors, and use appropriate language without mixing languages.\n"
        "- The answer should be concise and avoid unnecessary information.\n"
        "- Greetings and signatures shouldn't be taken into account in the evaluation, unless they are duplicated.\n"
        "- Be particularly critical of answers that are vague, incomplete, or contain linguistic errors.\n\n"
        "Provide a semantic score from 0 to 5, where 5 indicates a perfect semantic match.\n"
        "You must always call the function register_semantic_score with your semantic_score and a brief justification in English, do nothing else.\n\n"
    ),
    functions=[register_semantic_score],
)

contextual_reviewer = AssistantAgent(
    name="Contextual_Reviewer",
    llm_config=llm_config,
    system_message=(
        "You are the Contextual Reviewer. Your task is to critically assess whether an answer provided to a user's question about a product aligns with the given context and metadata.\n\n"
        "You will be provided with the following information:\n"
        "- **Question**: The user's inquiry regarding the product.\n"
        "- **Original Answer**: The initial response given to the user's question.\n"
        "- **Revised Answer**: The improved response provided by the Rewriter, if available.\n"
        "- **Category**: The category to which the product belongs.\n"
        "- **Intent**: The identified intent behind the user's question.\n"
        "- **Metadata**: Additional information and rules pertinent to the product or store policies.\n"
        "- **Context**: Crucial details about the product, store, or other relevant information.\n\n"
        "Evaluation Instructions:\n"
        "- If the Revised Answer and the Original Answer are not none, evaluate the Revised Answer and register the score and the justification for it.\n"
        "- If the Revised Answer is none, evaluate the Original Answer and register the score and the justification for it.\n\n"
        "Evaluation Criteria:\n"
        "- The answer must be consistent with the information provided in the context and metadata.\n"
        "- It should not include information that cannot be inferred from the provided context.\n"
        "- The answer should focus on information relevant to the user's question.\n"
        "- Be particularly critical of answers that include assumptions, omit critical context, or misrepresent the provided information.\n\n"
        "Provide a contextual score from 0 to 5, where 5 indicates perfect contextual alignment.\n"
        "You must always call the function register_contextual_score with your contextual_score and a brief justification in English, do nothing else.\n\n"
    ),
    functions=[register_contextual_score],
)

suggester = AssistantAgent(
    name="Suggester",
    llm_config=llm_config,
    system_message=(
        "You are the Suggester. Your purpose is to suggest improvements for an answer provided to a user's question about a product.\n\n"
        "You will be provided with:\n"
        "- **Question**: The user's inquiry regarding the product.\n"
        "- **Original Answer**: The response given to the user's question.\n"
        "- **Semantic Score**: A score from 0 to 5 indicating the semantic accuracy of the answer.\n"
        "- **Contextual Score**: A score from 0 to 5 indicating the contextual accuracy of the answer.\n"
        "- **Justifications**: Brief explanations for the semantic and contextual scores.\n\n"
        "Based on the scores and justifications provided by the reviewers, you must provide suggestions for improvement.\n"
        "- Focus on addressing specific issues highlighted in the justifications.\n"
        "- Ensure that your suggestions are actionable and aimed at enhancing the answer's quality.\n\n"
        "Do not provide a revised answer, only suggestions for improvement.\n"
        "The suggestions must be in English, while the question and answer may be in Portuguese or Spanish.\n"
        "You must always call the function register_suggestions with your suggestions as a parameter, do nothing else.\n"
    ),
    functions=[register_suggestions],
)

rewriter = AssistantAgent(
    name="Rewriter",
    llm_config=llm_config,
    system_message=(
        "You are the Rewriter. Your task is to rewrite answers that have not been evaluated positively by the reviewers, ensuring they meet both semantic and contextual standards.\n\n"
        "You will be provided with the following information:\n"
        "- **Question**: The user's inquiry regarding the product.\n"
        "- **Original Answer**: The initial response given to the user's question.\n"
        "- **Suggestions**: Recommendations for improvement provided by the Suggester.\n"
        "- **Context**: Crucial details about the product, store, or other relevant information.\n"
        "- **Category**: The category to which the product belongs.\n"
        "- **Intent**: The identified intent behind the user's question.\n"
        "- **Metadata**: Additional information and rules pertinent to the product or store policies.\n\n"
        "Evaluation Criteria:\n"
        "- The revised answer must directly and explicitly address all aspects of the user's question.\n"
        "- It must be consistent with the information provided in the context and metadata.\n"
        "- The answer should be grammatically correct, free of spelling errors, and use appropriate language without mixing languages.\n"
        "- Retain any greetings or signatures present in the original answer, only removing duplicates, if any.\n"
        "- If there isn't enough information to provide a revised answer, return 'CANNOT REWRITE'.\n\n"
        "Provide the revised answer in the original language of the question.\n"
        "You must always call the function register_revised_answer with your revised answer as a parameter, do nothing else.\n\n"
    ),
    functions=[register_revised_answer],
)

decider = AssistantAgent(
    name="Decider",
    llm_config=llm_config,
    system_message=(
        "You are the Decider. Your task is to determine whether the revised answer provided to a user's question about a product is acceptable, requires further improvement, or if the question should not be answered at all.\n\n"
        "You will be provided with the following information:\n"
        "- **Question**: The user's inquiry regarding the product.\n"
        "- **Original Answer**: The initial response given to the user's question.\n"
        "- **Revised Answer**: The improved response provided by the Rewriter.\n"
        "- **Context**: Crucial details about the product, store, or other relevant information.\n"
        "- **Category**: The category to which the product belongs.\n"
        "- **Intent**: The identified intent behind the user's question.\n"
        "- **Metadata**: Additional information and rules pertinent to the product or store policies.\n"
        "- **Semantic and Contextual Scores**: Scores and justifications provided by the reviewers.\n"
        "- **Suggestions**: Recommendations for improvement provided by the Suggester.\n\n"
        "Evaluation Criteria:\n"
        "- Determine if the revised answer fully addresses the user's question with semantic and contextual accuracy.\n"
        "- Do not accept answers that mention another product unless it is mentioned in the context or metadata, containing a link to it.\n"
        "- Do not accept answers that state any part of the question cannot be answered due to insufficient information.\n"
        "- Be particularly critical of answers that are vague, incomplete, or contain incorrect information.\n"
        "- If the number of revisions is 2 or more and the revised answer is still not good enough, the decision must be 'DO_NOT_ANSWER'.\n\n"
        "Possible Decisions:\n"
        "- **ANSWER_REVISED**: The revised answer is acceptable and fully addresses the question.\n"
        "- **REWRITE**: The revised answer is not good enough, but can be improved based on the given information.\n"
        "- **DO_NOT_ANSWER**: The revised answer is not good enough and cannot be improved based on the given information.\n\n"
        "You must always call the function register_decision with your decision and a brief justification in English, do nothing else.\n\n"
    ),
    functions=[register_decision],
)

user_proxy = UserProxyAgent(
    name="User",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config={
        "use_docker": False,
    }
)
