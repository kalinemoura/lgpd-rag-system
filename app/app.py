import streamlit as st
import os
from utils.save_docs import save_docs_to_vectordb
from utils.session_state import initialize_session_state_variables
from utils.prepare_vectordb import get_vectorstore
from utils.chatbot import chat


class ChatApp:
    """
    RAG Chatbot with fixed document base (LGPD)
    """

    def __init__(self):
        if not os.path.exists("docs"):
            os.makedirs("docs")

        # Streamlit config
        st.set_page_config(
            page_title="Consultor LGPD ⚖️", page_icon="⚖️", layout="centered"
        )

        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

            /* Aplicar fonte Inter em todo o app */
            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }

            /* Header fixo e compacto */
            .main-header {
                position: sticky;
                top: 0;
                z-index: 999;
                background: white;
                text-align: center;
                padding: 1.5rem 1rem 1rem;
                margin-bottom: 1.5rem;
                border-bottom: 2px solid #e0e0e0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }

            .main-title {
                font-family: 'Inter', sans-serif;
                font-size: 1.8rem;
                font-weight: 700;
                color: #1a1a1a;
                margin-bottom: 0.3rem;
                letter-spacing: -0.5px;
            }

            .badge {
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.3rem 0.8rem;
                border-radius: 16px;
                font-size: 0.7rem;
                font-weight: 600;
                margin: 0.3rem 0;
                box-shadow: 0 2px 6px rgba(102, 126, 234, 0.3);
            }

            .subtitle {
                font-family: 'Inter', sans-serif;
                font-size: 0.85rem;
                font-weight: 400;
                color: #666;
                margin-top: 0.5rem;
            }

            /* Ajustar padding do conteúdo principal */
            .block-container {
                padding-top: 0 !important;
            }

            /* Sidebar melhorado */
            .sidebar-content {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                margin-top: 1rem;
            }

            .sidebar-title {
                font-weight: 600;
                color: #333;
                margin-bottom: 0.5rem;
            }

            /* Melhorar input de chat */
            .stTextInput > div > div > input {
                border-radius: 12px;
                border: 2px solid #e0e0e0;
                padding: 0.75rem;
                font-size: 0.95rem;
            }

            .stTextInput > div > div > input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }

            /* Melhorar mensagens do chat */
            .stChatMessage {
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 0.75rem;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        # Header customizado (agora FIXO no topo)
        st.markdown(
            """
            <div class="main-header">
                <h1 class="main-title">Consultor LGPD ⚖️</h1>
                <span class="badge">⚡ Powered by RAG</span>
                <p class="subtitle">Fundamentado na Lei nº 13.709/2018</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        initialize_session_state_variables(st)

        if st.session_state.vectordb is None:
            if os.path.exists("Vector_DB - Documents"):
                st.session_state.vectordb = get_vectorstore(
                    os.listdir("docs"), from_session_state=True
                )
            else:
                st.session_state.vectordb = get_vectorstore(
                    os.listdir("docs"), from_session_state=False
                )

    def run(self):
        #sidebar melhorado
        with st.sidebar:
            st.markdown("### 📚 Base de Conhecimento")

            docs = os.listdir("docs")
            if docs:
                st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
                st.markdown("**Documentos carregados:**")
                for doc in docs:
                    st.markdown(f"📄 {doc}")
                st.markdown("</div>", unsafe_allow_html=True)

                #informações adicionais
                st.markdown("---")
                st.markdown("### ℹ️ Sobre")
                st.info(
                    "Este chatbot utiliza **RAG (Retrieval-Augmented Generation)** "
                    "para fornecer respostas fundamentadas na Lei Geral de Proteção "
                    "de Dados (LGPD - Lei nº 13.709/2018)."
                )
            else:
                st.warning("⚠️ Nenhum documento encontrado em /docs")

            st.markdown("---")
            st.markdown("### ⚙️ Modo de resposta")
            mode_label = st.radio(
                "Pipeline",
                options=["V2 — RAG clássico (vetorial)", "V3 — GraphRAG"],
                label_visibility="collapsed",
            )
            mode = "v3" if mode_label.startswith("V3") else "v2"

            graph = None
            if mode == "v3":
                graph = self._load_graph()
                if graph is None:
                    st.error(
                        "Grafo não encontrado. Rode `python -m utils.graphrag.build_graph` "
                        "primeiro para gerá-lo (esse passo chama a API da OpenAI uma vez "
                        "e cacheia o resultado em `graph_cache/`)."
                    )

        if st.session_state.vectordb is None:
            st.error("❌ Vector database não foi inicializado.")
            return

        if mode == "v3" and graph is None:
            return  # já mostrou o erro na sidebar, não deixa conversar sem grafo

        st.session_state.chat_history = chat(
            st.session_state.chat_history,
            st.session_state.vectordb,
            mode=mode,
            graph=graph,
        )

    @staticmethod
    def _load_graph():
        from utils.graphrag.graph_builder import load_graph

        if "kg_graph" not in st.session_state:
            try:
                st.session_state.kg_graph = load_graph()
            except FileNotFoundError:
                st.session_state.kg_graph = None
        return st.session_state.kg_graph


if __name__ == "__main__":
    app = ChatApp()
    app.run()
