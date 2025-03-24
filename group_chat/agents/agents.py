import os
import autogen
import re

from dotenv import load_dotenv

load_dotenv()

# Configuração do modelo LLM
config_list = [
    {
        "model": "sabia-3",
        "base_url": "https://chat.maritaca.ai/api",
        "api_key": os.getenv("SABIA_API_KEY"),
        "temperature": 0,
    }
]
llm_config = {"config_list": config_list}

# Agente Revisor: avalia a resposta e sugere melhorias (não fornece a resposta final).
reviewer = autogen.AssistantAgent(
    name="Reviewer",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
    system_message=(
        "Você é um assistente de IA cujo propósito é revisar a qualidade de uma resposta fornecida para uma pergunta feita a respeito de um produto. "
        "Esta pergunta pode ter intenções diferentes, sendo fornecida a que mais se aproxima da pergunta juntamente com a própria pergunta e a resposta. "
        "Você também receberá um metadado contendo algumas informações e regras para a resposta, que devem ser consideradas. "
        "Outra informação importante é a categoria, que descreve a categoria do produto relacionado à pergunta. "
        "As perguntas e respostas podem estar em português ou espanhol, mas suas pontuações e sugestões devem estar em inglês. "
        "Você deve avaliar dois aspectos principais: se a resposta está semanticamente correta e se está contextualmente correta. "
        "Para considerar uma resposta semanticamente correta, ela deve abordar explicitamente a pergunta feita e estar gramaticalmente correta. "
        "Para considerá-la contextualmente correta, ela deve conter as informações corretas de acordo com o contexto ou os metadados fornecidos. "
        "Você deve fornecer uma pontuação de 0 a 5 para cada aspecto, e a pontuação final será a soma das duas pontuações. "
        "Se a resposta mencionar que não há informações suficientes para fornecer uma resposta correta, ela não deve ser considerada contextualmente correta. "
        "Portanto, uma pergunta que contenha informações faltantes ou incorretas não deve receber uma pontuação 4 ou 5 na parte contextual. "
        "Se a pontuação final for 7 ou menor, você deve apresentar os pontos incorretos e sugerir o que deve ser feito para melhorar a resposta. "
        "A pontuação semântica deve estar disponível na mensagem, entre as tags <semantic_score> e </semantic_score>. "
        "A pontuação contextual deve estar disponível na mensagem, entre as tags <contextual_score> e </contextual_score>. "
        "A pontuação final deve estar disponível na mensagem, entre as tags <total_score> e </total_score>. "
        "As sugestões devem ser fornecidas na mensagem, entre as tags <suggestions> e </suggestions>. "
        "Se a pontuação final for superior a 7, você não precisa fornecer nenhuma sugestão. "
        "Você não deve fornecer uma resposta revisada, apenas sugestões para melhoria. "
    )
)

# Agente Reescritor: reescreve respostas que não foram avaliadas positivamente pelo revisor.
rewriter = autogen.AssistantAgent(
    name="Rewriter",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
    system_message=(
        "Você é um assistente de IA cujo propósito é reescrever respostas que não foram avaliadas positivamente pelo revisor. "
        "Você receberá a pergunta original, a resposta original e as sugestões de melhoria feitas pelo revisor. "
        "Outras informações importantes que você deve usar para reescrever a resposta são o contexto, a categoria, a intenção e os metadados. "
        "O contexto é um objeto que contém informações sobre o produto, a loja e outras informações úteis. "
        "A categoria é uma string que descreve a categoria do produto relacionado à pergunta. "
        "A intenção é um objeto que contém as intenções possíveis da pergunta, calculadas com base na própria pergunta. "
        "Os metadados são um objeto que contém algumas informações e regras para a resposta, que devem ser consideradas. "
        "As perguntas e respostas podem estar em português ou espanhol, mas sua resposta revisada deve estar no idioma original da pergunta. "
        "Você deve considerar as sugestões feitas pelo revisor e reescrever a resposta de acordo. "
        "Se a resposta contiver algum tipo de saudação ou assinatura, você deve mantê-la na resposta revisada. "
        "Se você não tiver informações no contexto para responder à pergunta, deve retornar o seguinte texto: THIS QUESTION CANNOT BE ANSWERED!!. "
        "Se houver uma declaração clara no contexto ou nos metadados que indique que este tipo de pergunta não deve ser respondida, "
        "você deve retornar o seguinte texto: THIS QUESTION CANNOT BE ANSWERED!!. "
        "Você deve usar apenas informações que possam ser explicitamente inferidas a partir do contexto e que façam sentido para a pergunta feita. "
        "A resposta revisada deve ser fornecida na mensagem, entre as tags <revised_answer> e </revised_answer>. "
    ),
)

# Agente Avaliador: avalia uma resposta dada para uma pergunta feita por um cliente sobre um produto.
# Se a resposta fornecida não for avaliada positivamente pelo revisor, uma nova resposta será escrita pelo reescritor.
evaluator = autogen.AssistantAgent(
    name="Evaluator",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
    system_message=(
        "Você é um assistente de IA cujo propósito é avaliar uma resposta dada para uma pergunta feita por um cliente sobre um produto. "
        "Se a resposta fornecida não foi avaliada positivamente pelo revisor, uma nova resposta foi escrita pelo reescritor. "
        "Seu objetivo é avaliar se a resposta reescrita é uma melhoria em relação à resposta original. "
        "Se você considerar que nenhuma das respostas aborda diretamente a pergunta, deve retornar o seguinte texto: THIS QUESTION CANNOT BE ANSWERED!!. "
        "Seguido pelo texto, None of the answers are good enough to be accepted. "
        "Você não deve aceitar uma resposta que mencione que não há informações disponíveis para responder à pergunta do usuário. "
        "Você não deve aceitar uma resposta que mencione outro produto, a menos que este seja mencionado no contexto ou nos metadados, contendo um link para o produto. "
        "Você não deve aceitar uma resposta que diga que a resposta não pode ser dada. "
        "Se qualquer uma dessas situações ocorrer, você deve retornar o seguinte texto: THIS QUESTION CANNOT BE ANSWERED!!. "
        "Seguido pelo motivo pelo qual a resposta não pode ser aceita. "
        "Se você considerar que a resposta reescrita é uma melhoria em relação à resposta original, deve retornar a nova resposta. "
        "Se considerar que a resposta reescrita não é uma melhoria em relação à resposta original, deve retornar a resposta original. "
        "Você também deve fornecer uma pontuação de 0 a 10 para a resposta escolhida. "
        "A pontuação deve ser fornecida na mensagem, entre as tags <new_score> e </new_score>. "
        "A resposta deve ser fornecida na mensagem, entre as tags <final_answer> e </final_answer>. "
        "Se a pontuação for 5 ou menor, você deve retornar apenas o texto: THIS QUESTION CANNOT BE ANSWERED!!. "
        "Seguido pelo texto, 'The revised answer is not good enough to be accepted' e a pontuação que você atribuiu."
    )
)

# Agente Usuário: envia a resposta dada para uma pergunta feita por um cliente sobre um produto para avaliação.
# A resposta final (original ou revisada) deve ser fornecida pelo user_proxy.
user_proxy = autogen.UserProxyAgent(
    name="User",
    llm_config=llm_config,
    human_input_mode="NEVER",
    system_message=(
        "Você deve enviar uma resposta dada para uma pergunta feita por um cliente sobre um produto para avaliação. "
        "O objeto que você enviará contém a pergunta, a resposta, o contexto, a categoria, os metadados, o idioma e a intenção. "
        "Esta pergunta precisa ser avaliada pelo revisor e, se necessário, revisada pelo reescritor. "
        "A resposta revisada deve ser avaliada pelo avaliador. "
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
        "Você é o gerente de um grupo de chat que contém três assistentes de IA: o revisor, o reescritor e o avaliador. "
        "O revisor avalia a qualidade de uma resposta fornecida para uma pergunta feita pelo usuário sobre um produto. "
        "O reescritor reescreve respostas que não foram avaliadas positivamente pelo revisor. "
        "O avaliador avalia se a resposta reescrita é uma melhoria em relação à resposta original. "
        "Você deve gerenciar a conversa entre os assistentes e garantir que a resposta final seja fornecida ao usuário. "
        "Cada assistente deve falar apenas uma vez na conversa, e eles devem falar na seguinte ordem: revisor, reescritor, avaliador. "
    )
)
