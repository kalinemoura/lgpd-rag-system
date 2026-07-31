import os
from utils.prepare_vectordb import get_vectorstore


def initialize_session_state_variables(st):
    """
    Initialize session state variables for the Streamlit application

    Parameters:
    - st (streamlit.delta_generator.DeltaGenerator): Streamlit's DeltaGenerator object used for rendering elements
    """
    # Get the list of uploaded documents (só .pdf — evita tentar processar
    # imagens ou outros arquivos que estejam soltos na pasta 'docs')
    upload_docs = [f for f in os.listdir("docs") if f.lower().endswith(".pdf")]
    # List of session state variables to initialize
    variables_to_initialize = [
        "chat_history",
        "uploaded_pdfs",
        "processed_documents",
        "vectordb",
        "previous_upload_docs_length",
    ]
    # Iterate over the variables and initializes them if not present in the session state
    for variable in variables_to_initialize:
        if variable not in st.session_state:
            if variable == "processed_documents":
                # Set to the name of the files present in the docs folder
                st.session_state.processed_documents = upload_docs
            elif variable == "vectordb":
                if not upload_docs:
                    st.session_state.vectordb = None
                elif os.path.exists("Vector_DB - Documents"):
                    # Banco já existe -> só carrega (rápido)
                    st.session_state.vectordb = get_vectorstore(
                        upload_docs, from_session_state=True
                    )
                else:
                    # Banco não existe (primeira vez, ou foi apagado) ->
                    # reconstrói do zero automaticamente. Pode demorar um
                    # pouco mais nesse primeiro start.
                    with st.spinner(
                        "Vector DB não encontrado — reconstruindo a partir dos PDFs em 'docs'..."
                    ):
                        st.session_state.vectordb = get_vectorstore(
                            upload_docs, from_session_state=False
                        )
            elif variable == "previous_upload_docs_length":
                # Set to the quantity of documents in the docs folder during app startup
                st.session_state.previous_upload_docs_length = len(upload_docs)
            else:
                st.session_state[variable] = []
