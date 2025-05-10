import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Carregar variáveis de ambiente do .env
load_dotenv()

# Configuração da página (deve ser a primeira chamada do Streamlit)
st.set_page_config(page_title="Gym Assistant Chat", layout="wide")

# CSS personalizado para um visual mais sóbrio
st.markdown(
    """
    <style>
    .stApp { background-color: #f4f4f4; color: #333333; }
    .css-1d391kg { font-family: 'Helvetica Neue', Arial, sans-serif; }
    .stChatMessage { border-radius: 0; border: 1px solid #ccc; padding: 8px; background-color: #fff; }
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
        st.session_state.conversation_memory = ConversationBufferMemory()
    return st.session_state.conversation_memory

memory = get_memory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# Cabeçalho com avatar
col1, col2 = st.columns([1, 10])
with col1:
    if os.path.exists(environment_avatar): st.image(environment_avatar, width=60)
    else: st.write(":robot_face:")
with col2:
    st.header("Gym Assistant Chat")

# Iniciar histórico com mensagem de encaminhamento e primeira pergunta
if "history" not in st.session_state:
    st.session_state.history = []
    intro = (
        "Olá, sou o Gym Assistant, seu assistente profissional. Antes de começarmos, gostaria de entender melhor seu contexto."
    )
    st.session_state.history.append(("assistant", intro))
    # Pergunta inicial para guiar a conversa
    first_q = "Qual é seu principal objetivo hoje? (treino, nutrição, suplementação ou organização)"
    st.session_state.history.append(("assistant", first_q))

# Exibir histórico de mensagens
for role, msg in st.session_state.history:
    if role == "assistant":
        st.chat_message(role, avatar=environment_avatar if os.path.exists(environment_avatar) else None).write(msg)
    else:
        st.chat_message(role).write(msg)

# Capturar entrada do usuário
user_input = st.chat_input("Digite sua resposta...")
if user_input:
    # Salvar e exibir input
    st.session_state.history.append(("user", user_input))
    st.chat_message("user").write(user_input)
    # Gerar próxima resposta com base no contexto
    with st.spinner("Aguarde..."):
        resposta = conversation.predict(input=user_input)
    st.session_state.history.append(("assistant", resposta))
    st.chat_message("assistant", avatar=environment_avatar if os.path.exists(environment_avatar) else None).write(resposta)
