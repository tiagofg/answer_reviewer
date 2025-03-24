import os
import autogen
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
    max_consecutive_auto_reply=2,
    is_termination_msg=lambda msg: "It is not possible to provide a revised answer." in msg.get("content", ""),
    system_message=(
        "Você é um assistente de IA cujo propósito é revisar a qualidade de uma resposta fornecida "
        "para uma pergunta feita ao usuário sobre um produto. Esta pergunta pode ter 3 intenções diferentes: "
        "1. Compatibilidade: o usuário pergunta se o produto é compatível com outro produto. "
        "2. Especificação: o usuário pergunta sobre as especificações do produto. "
        "3. Disponibilidade: o usuário pergunta sobre a disponibilidade do produto. "
        "As perguntas e respostas podem estar em português ou espanhol, mas suas pontuações e sugestões devem estar em inglês. "
        "Você deve avaliar dois aspectos principais: se a resposta está semanticamente correta e se está contextualmente correta. "
        "Junto com a pergunta e a resposta, será fornecido um contexto que deve ser considerado para a avaliação. "
        "Você deve fornecer uma pontuação de 0 a 5 para cada aspecto, e a pontuação final será a soma das duas pontuações. "
        "Para cada pontuação, você deve colocar o número da pontuação seguido de uma barra e da pontuação máxima possível. "
        "A nota para resposta deve estar logo após o texto 'Final Score:', onde você deve apresentar a pontuação seguido de /10. "
        "Se a pontuação final for 7 ou menor, você deve apresentar os pontos incorretos e sugerir o que deve ser feito para melhorar a resposta. "
        "As sugestões devem estar no final da mensagem, após o texto 'Suggestions:'. "
        "Se a pontuação final for superior a 7, você não precisa fornecer sugestões. "
        "Você não deve fornecer uma resposta revisada, o usuário fará as correções necessárias e retornará a resposta corrigida para avaliação. "
    )
)

# Agente Usuário: envia a pergunta para avaliação e, se necessário, revisa a resposta de acordo com as sugestões do revisor.
# A resposta final (original ou revisada) deve ser fornecida por user_proxy.
user_proxy = autogen.UserProxyAgent(
    name="User",
    llm_config=llm_config,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    is_termination_msg=lambda msg: "Final Score" in msg.get("content", "") and int(
        msg.get("content", "").split("Final Score: ")[1].split("/")[0]
    ) > 7,
    system_message=(
        "Você deve enviar um conjunto de perguntas e respostas para serem avaliadas por um assistente de IA. "
        "Cada pergunta pode ter 3 intenções diferentes: "
        "1. Compatibilidade: o usuário pergunta se o produto é compatível com outro produto. "
        "2. Especificação: o usuário pergunta sobre as especificações do produto. "
        "3. Disponibilidade: o usuário pergunta sobre a disponibilidade do produto. "
        "As perguntas e respostas podem estar em português ou espanhol; ao reescrever a resposta, você deve considerar o idioma original da pergunta. "
        "Se a pontuação final fornecida pelo revisor for inferior a 6, ele apresentará os pontos incorretos e sugerirá o que deve ser feito para melhorar a resposta. "
        "Você deve fazer as correções sugeridas e retornar a resposta corrigida para ser avaliada novamente. "
        "Imediatamente após o texto 'Revised Answer:', você deve fornecer a resposta corrigida. Isso deve ser o final da mensagem. "
        "Se você acreditar que, dado o contexto, não é possível fornecer uma resposta correta, você deve retornar o seguinte texto: "
        "'It is not possible to provide a revised answer.' "
        "Se a resposta contiver algum tipo de saudação ou assinatura, você deve mantê-la na resposta revisada. "
    ),
    code_execution_config={
        "use_docker": False,
    }
)
