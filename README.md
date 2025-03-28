# Projeto Chatbot de Informações sobre Cidades Brasileiras

## Índice
1. [Sobre o Projeto](#sobre-o-projeto)
2. [Como Clonar o Repositório](#como-clonar-o-repositorio)
3. [Instalação de Dependências](#instalacao-de-dependencias)
4. [Explicação do Código](#explicacao-do-codigo)
5. [Glossário](#glossario)
6. [Contribuição](#contribuicao)
7. [Licença](#licenca)

---

## Sobre o Projeto

Este projeto implementa um chatbot que fornece informações sobre cidades brasileiras, como população, pontos turísticos e universidades. Ele utiliza a API da Groq e o framework LangChain para processar as perguntas do usuário e fornecer respostas.

---

## Como Clonar o Repositório

Para copiar este repositório para sua máquina local, siga os seguintes passos:

1. Abra o terminal ou prompt de comando.
2. Navegue até o diretório onde deseja salvar o projeto.
3. Execute o seguinte comando:
   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
   ```
4. Acesse a pasta do projeto:
   ```bash
   cd nome-do-repositorio
   ```

Caso não tenha o `git` instalado, você pode baixar o projeto manualmente pelo botão "Download ZIP" no GitHub.

---

## Instalação de Dependências

Este projeto possui dependências listadas no arquivo `requirements.txt`. Para instalá-las, siga os passos abaixo:

1. Certifique-se de que tem o Python instalado (versão X.X ou superior).
2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Para Linux/macOS
   venv\Scripts\activate  # Para Windows
   ```
3. Instale as dependências com o comando:
   ```bash
   pip install -r requirements.txt
   ```

Agora, todas as bibliotecas necessárias estão instaladas e você pode rodar o projeto.

---

## Explicação do Código

### 1. Importação das Bibliotecas

```python
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
```

Aqui, carregamos bibliotecas essenciais para manipular variáveis de ambiente, gerenciar memória de conversação e processar mensagens do chatbot.

### 2. Carregamento de Variáveis de Ambiente

```python
load_dotenv(find_dotenv())
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key
```

Essas linhas garantem que a chave da API do Groq seja carregada corretamente do arquivo `.env`.

### 3. Banco de Dados com Informações das Cidades

```python
city_data = {
    "São Paulo": {"população": "12,33 milhões", "pontos_turisticos": [...], "universidade": "USP"},
    "Rio de Janeiro": {"população": "6,7 milhões", "pontos_turisticos": [...], "universidade": "UFRJ"},
    ...
}
```

Aqui estão armazenadas informações sobre diversas cidades brasileiras.

### 4. Configuração do Modelo de IA

```python
llm = ChatGroq(temperature=0.7, model="gemma2-9b-it")
```

Definição do modelo de IA usado para gerar respostas.

### 5. Configuração do Prompt Template

```python
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("Você é um assistente de IA..."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{question}")
])
```

Esse trecho define como o chatbot interage com o usuário.

### 6. Gerenciamento de Memória de Conversação

```python
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
```

Mantém o histórico de conversações.

### 7. Função de Busca de Informações sobre Cidades

```python
def get_city_info(city_name, info_type):
    ...
```

Verifica e retorna informações específicas sobre uma cidade.

### 8. Função Principal do Chatbot

```python
def chatbot_response(question):
    ...
```

Identifica a cidade e o tipo de informação desejada, gerando uma resposta.

### 9. Exemplo de Uso

```python
print(chatbot_response("Quais as principais atrações de São Paulo?"))
```

Exemplo de interação com o chatbot.

---

## Glossário

- **LLM (Large Language Model)**: Modelo de IA capaz de processar linguagem natural.
- **API**: Interface para comunicação entre diferentes sistemas.
- **Memória de Conversação**: Sistema que armazena interações anteriores.
- **Template de Prompt**: Configuração do formato das perguntas e respostas.
- **Variáveis de Ambiente**: Valores sensíveis armazenados separadamente do código-fonte.

---

## Contribuição

Se quiser contribuir com este projeto:

1. Fork o repositório.
2. Crie uma branch para suas mudanças.
3. Submeta um Pull Request.

---

## Licença

Este projeto está licenciado sob a [Nome da Licença]. Consulte o arquivo `LICENSE` para mais detalhes.

