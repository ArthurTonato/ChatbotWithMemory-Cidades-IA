import os
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough

load_dotenv(find_dotenv())

# Dados das cidades
city_data = {
    "São Paulo": {
        "população": "12,33 milhões",
        "pontos_turisticos": ["Parque Ibirapuera", "Avenida Paulista", "Mercado Municipal", "Catedral da Sé"],
        "universidade": "Universidade de São Paulo (USP)"
    },
    "Rio de Janeiro": {
        "população": "6,7 milhões",
        "pontos_turisticos": ["Cristo Redentor", "Pão de Açúcar", "Praia de Copacabana"],
        "universidade": "Universidade Federal do Rio de Janeiro (UFRJ)"
    },
    "Salvador": {
        "população": "2,9 milhões",
        "pontos_turisticos": ["Pelourinho", "Elevador Lacerda", "Farol da Barra"],
        "universidade": "Universidade Federal da Bahia (UFBA)"
    },
    "Belo Horizonte": {
        "população": "2,5 milhões",
        "pontos_turisticos": ["Praça da Liberdade", "Igreja São José", "Museu de Artes e Ofícios"],
        "universidade": "Universidade Federal de Minas Gerais (UFMG)"
    },
    "Fortaleza": {
        "população": "2,7 milhões",
        "pontos_turisticos": ["Praia do Futuro", "Catedral Metropolitana", "Mercado Central"],
        "universidade": "Universidade Federal do Ceará (UFC)"
    },
    "Brasília": {
        "população": "3,1 milhões",
        "pontos_turisticos": ["Congresso Nacional", "Catedral de Brasília", "Palácio do Planalto"],
        "universidade": "Universidade de Brasília (UnB)"
    },
    "Curitiba": {
        "população": "1,9 milhões",
        "pontos_turisticos": ["Jardim Botânico", "Ópera de Arame", "Rua XV de Novembro"],
        "universidade": "Universidade Federal do Paraná (UFPR)"
    },
    "Porto Alegre": {
        "população": "1,5 milhões",
        "pontos_turisticos": ["Parque Redenção", "Caminho dos Antiquários", "Fundação Ibere Camargo"],
        "universidade": "Universidade Federal do Rio Grande do Sul (UFRGS)"
    },
    "Recife": {
        "população": "1,6 milhões",
        "pontos_turisticos": ["Praia de Boa Viagem", "Instituto Ricardo Brennand", "Marco Zero"],
        "universidade": "Universidade Federal de Pernambuco (UFPE)"
    },
    "Manaus": {
        "população": "2,1 milhões",
        "pontos_turisticos": ["Teatro Amazonas", "Encontro das Águas", "Palácio Rio Negro"],
        "universidade": "Universidade Federal do Amazonas (UFAM)"
    },
    "Natal": {
        "população": "1,4 milhões",
        "pontos_turisticos": ["Forte dos Reis Magos", "Praia de Ponta Negra", "Dunas de Genipabu"],
        "universidade": "Universidade Federal do Rio Grande do Norte (UFRN)"
    },
    "Maceió": {
        "população": "1,0 milhão",
        "pontos_turisticos": ["Praia do Francês", "Palácio Marechal Floriano Peixoto", "Igreja de São Gonçalo do Amarante"],
        "universidade": "Universidade Federal de Alagoas (UFAL)"
    },
    "Cuiabá": {
        "população": "620 mil",
        "pontos_turisticos": ["Parque Nacional de Chapada dos Guimarães", "Catedral Basílica do Senhor Bom Jesus", "Museu do Morro da Caixa D'Água"],
        "universidade": "Universidade Federal de Mato Grosso (UFMT)"
    },
    "Aracaju": {
        "população": "650 mil",
        "pontos_turisticos": ["Praia de Atalaia", "Museu Palácio Marechal Floriano Peixoto", "Mercado Municipal"],
        "universidade": "Universidade Federal de Sergipe (UFS)"
    }
}

# Configuração da API Key do Groq
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key

# Inicialização do modelo Groq
llm = ChatGroq(temperature=0.7, model="gemma2-9b-it")

# Configuração do prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Você é um assistente de IA que fornece informações sobre cidades brasileiras."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{question}")
])

# Inicialização do histórico de mensagens
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Criação da cadeia LLM
chain = {
    "question": RunnablePassthrough(),
    "history": lambda x: memory.load_memory_variables({})["history"]
} | prompt | llm

def get_city_info(city_name, info_type):
    """Retorna informações específicas sobre a cidade."""
    if city_name in city_data:
        city_info = city_data[city_name]
        if info_type == "população":
            return f"População: {city_info['população']}"
        elif info_type == "pontos turísticos":
            return f"Pontos Turísticos: {', '.join(city_info['pontos_turisticos'])}"
        elif info_type == "universidade":
            return f"Universidade: {city_info['universidade']}"
        else:
            return "Desculpe, não entendi qual informação você deseja."
    else:
        return "Desculpe, não tenho informações sobre esta cidade."

def chatbot_response(question):
    """Gera uma resposta do chatbot para a pergunta do usuário."""
    city_name = None
    info_type = None
    question_lower = question.lower().strip()

    for city in city_data:
        city_lower = city.lower().strip()
        if city_lower in question_lower:
            city_name = city
            break

    if "população" in question_lower:
        info_type = "população"
    elif "pontos turísticos" in question_lower or "atrações" in question_lower:
        info_type = "pontos turísticos"
    elif "universidade" in question_lower:
        info_type = "universidade"

    if city_name and info_type:
        info = get_city_info(city_name, info_type)
        response = chain.invoke({"question": info})
        memory.save_context({"input": question}, {"output": response.content})
        return response.content
    else:
        return "Desculpe, não consigo responder a essa pergunta."

# Exemplo de interação com o chatbot
print(chatbot_response("Quais as principais atrações de São Paulo?"))