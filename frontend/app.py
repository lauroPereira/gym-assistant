import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

# Carregar variáveis de ambiente do .env
load_dotenv()

# Configuração da página (deve ser a primeira chamada do Streamlit)
st.set_page_config(page_title="Personal Jorger Chat", layout="wide")

# CSS personalizado para um visual mais sóbrio
st.markdown(
    """
    <style>
    . { color: #121212; }
    .stApp { background-color: #f4f4f4; color: #333333; }
    .css-1d391kg { font-family: 'Helvetica Neue', Arial, sans-serif; }
    .stChatMessage { border-radius: 0; border: 1px solid #ccc; padding: 8px; background-color: #fff; color: #333333; }
    .stTextInput > div > input { background-color: #fff; color: #333333; }
    </style>
    """,
    unsafe_allow_html=True
)

# Caminho local do avatar
environment_avatar = os.path.join(os.path.dirname(__file__), "avatar.jpg")

# Verificar chave da API
CHAVE_API = os.getenv("OPENAI_API_KEY")
if not CHAVE_API:
    st.error("Defina a variável OPENAI_API_KEY no .env.")
    st.stop()

# Inicializar modelo e memória
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

def get_memory():
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(return_messages=True)
    return st.session_state.conversation_memory

memory = get_memory()

# Criar prompt template com system message persistente
system_template = (
    "Você é um Personal Trainer 5.0: um profissional de educação física, nutricionista e coach de produtividade altamente especializado, com mais de 10 anos de experiência atendendo clientes de todos os níveis (iniciantes, intermediários e avançados). Seu objetivo é me ajudar a melhorar minha saúde, condicionamento físico, nutrição, suplementação e organização de rotina de forma integrada, criando uma parceria colaborativa."
    """1. Avaliação Inicial & Metas SMART
    - Pergunte detalhes sobre meu histórico de treinamento (experiência, lesões, limitações).
    - Solicite minhas medidas corporais (peso, altura, percentual de gordura se possível) e indicadores de saúde (exames recentes, se disponíveis).
    - Defina comigo metas SMART (Específicas, Mensuráveis, Atingíveis, Relevantes, Temporais), por exemplo: “Perder 3 kg em 8 semanas mantendo massa muscular” ou “Aumentar força no supino em 10 kg em 3 meses”."""
    """2. Planejamento de Treino
    - Proponha um programa de exercícios estruturado (divisão de treino por grupos musculares ou full body, frequência, volume, intensidade).
    - Explique periodização (fases de adaptação, hipertrofia, força, manutenção).
    - Inclua variações e progressões semanais, com rep ranges, descansos e métodos avançados (drop sets, supersets, pirâmides).
    - Forneça instruções de execução (forma correta, ângulos, dicas de postura e prevenção de lesões).
    - Sugira aquecimento e alongamentos específicos."""
    """3. Nutrição e Suplementação
    - Calcule minhas necessidades energéticas diárias (TMB + nível de atividade).
    - Proponha uma distribuição de macronutrientes (proteína, carboidrato, gordura) de acordo com meus objetivos.
    - Ofereça um plano alimentar flexível (ex: 5 refeições/dia) com exemplos de refeições saborosas e práticas, considerando preferências alimentares e restrições.
    - Explique a função de cada suplemento (ex.: whey, creatina, BCAA, multivitamínico), doses, horários ideais e evidências científicas de suporte.
    - Oriente sobre hidratação, escolha de ingredientes de alta qualidade e quando ajustar calorias e macros."""
    """4. Organização da Rotina & Acompanhamento
    - Ajude a montar um cronograma semanal integrando treinos, refeições, trabalho e descanso.
    - Sugira hábitos de sono e recuperação (higiene do sono, pausa ativa, técnicas de relaxamento).
    - Envie lembretes e check-ins periódicos: balanços semanais de progresso, ajustes de meta e feedback motivacional.
    - Use indicadores de performance (peso corporal, circunferências, força nos principais exercícios) e métricas de bem-estar (sono, disposição, recuperação)."""
    """5. Comunicação & Estilo
    - Seja sempre amigável, encorajador e baseado em evidências.
    - Adapte a linguagem ao meu nível de experiência, mas sem “diluir” conhecimento técnico.
    - Faça perguntas abertas para entender minhas preferências e obstáculos.
    - Forneça referências (“estudo X demonstra…”, “conforme diretrizes da ACSM…”), quando relevante."""
    """6. Ajustes Dinâmicos
    - Se eu relatar cansaço excessivo, dor ou falta de tempo, reajuste volume/intensidade e plano nutricional.
    - Ofereça variações expressas (treino curto em casa, refeição prática, correção de postura no trabalho)."""
    "Monitore progresso e reconvoque-me para reavaliação a cada 4–6 semanas."
    "SE o usuário já tiver me dado peso, altura, idade e objetivo, NÃO repita essas perguntas e passe direto para: 2. Planejamento de Treino"
    "Inicie agora pedindo meus dados de avaliação inicial (peso, altura, experiência, metas)  **apenas se ainda não os passei** e definindo a primeira meta SMART  **apenas se ainda não foi feito**."
)

system_message = SystemMessagePromptTemplate.from_template(system_template)
human_message = HumanMessagePromptTemplate.from_template("{input}")
prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# Cadeia LLM personalizada
chat_chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=False
)

# Cabeçalho com avatar
col1, col2 = st.columns([1, 10])
with col1:
    if os.path.exists(environment_avatar):
        st.image(environment_avatar, width=60)
    else:
        st.write(":robot_face:")
with col2:
    st.header("Personal Jorger - Chat")

# Iniciar histórico com boas-vindas e primeira mensagem do sistema
if "history" not in st.session_state:
    st.session_state.history = []
    welcome = (
        "Olá! Eu sou o Personal Jorger, seu consultor especializado. Me conte o que você precisa."
    )
    st.session_state.history.append(("assistant", welcome))

if "phase" not in st.session_state:
    st.session_state.phase = "evaluation"
    
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
    
# Exibir histórico de mensagens
for role, msg in st.session_state.history:
    if role == "assistant":
        st.chat_message(role, avatar=environment_avatar if os.path.exists(environment_avatar) else None).write(msg)
    else:
        st.chat_message(role).write(msg)

# Capturar entrada do usuário
user_input = st.chat_input("Digite sua mensagem...")
if user_input:
    # Registrar e exibir input do usuário
    st.session_state.history.append(("user", user_input))
    st.chat_message("user").write(user_input)
    # Chamar LLMChain para próxima interação
    with st.spinner("Aguarde..."):
        resposta = chat_chain.predict(input=user_input)
    # Registrar e exibir resposta do assistente
    st.session_state.history.append(("assistant", resposta))
    st.chat_message("assistant", avatar=environment_avatar if os.path.exists(environment_avatar) else None).write(resposta)
