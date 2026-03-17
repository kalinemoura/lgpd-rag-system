import streamlit as st
from collections import defaultdict

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

DEBUG_RETRIEVAL = False


def get_context_retriever_chain(vectordb):
    load_dotenv()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        # convert_system_message_to_human=True
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    # DEBUG TEMPORÁRIO DE RETRIEVAL
    # test_question = "Em quais situações o consentimento não é necessário?"

    # docs = retriever.get_relevant_documents(test_question)

    # print("\n=== DEBUG RETRIEVAL ===")
    # for i, doc in enumerate(docs):
    #     print(f"\nDoc {i}")
    #     print("Page:", doc.metadata.get("page"))
    #     print("Source:", doc.metadata.get("source"))
    #     print("Preview:", doc.page_content[:300])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Você é um assistente especializado na Lei Geral de Proteção de Dados (LGPD) que responde perguntas com  base EXCLUSIVA no contexto fornecido.\n"
            "Use apenas as informações do contexto. Não utilize conhecimento externo.\n"
            "Não invente fatos, definições ou definições.\n"
            "Responda sempre em português do Brasil.\n\n"
            "Se a resposta estiver claramente no contexto, responda de forma objetiva.\n"
            "Se houver apenas informação parcial, responda apenas com o que estiver sustentado no texto.\n"
            "Se não houver informação suficiente, diga claramente que a informação solicitada não foi localizada nos trechos recuperados da LGPD.\n\n"
            "Contexto:\n{context}\n\n"
            "Pergunta:\n{question}\n\n"
            "Resposta:"
        ),
    )

    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return retrieval_chain


def get_response(question, chat_history, vectordb):
    # Monta o chain
    chain = get_context_retriever_chain(vectordb)

    # Debug opcional
    if DEBUG_RETRIEVAL:
        retriever = vectordb.as_retriever(search_kwargs={"k": 10})
        docs = retriever.get_relevant_documents(question)

        # print("\n=== DEBUG RETRIEVAL ===")
        # for i, doc in enumerate(docs):
        #     print(f"\nDoc {i}")
        #     print("Page:", doc.metadata.get("page"))
        #     print("Source:", doc.metadata.get("source"))
        #     print("Preview:", doc.page_content[:300])

    # Executa chain
    try:
        response = chain.invoke(question)
    except Exception as e:
        print("\n=== ERRO TÉCNICO ===")
        print(str(e))
        return "Ocorreu um erro interno. Tente novamente.", []

    answer = response["result"]
    source_docs = response["source_documents"]

    # DEBUG – Inspeção da ordenação semântica do top_k
    print("\n=== DEBUG ORDEM SEMÂNTICA ===")
    for i, doc in enumerate(source_docs):
        print(i, doc.metadata.get("article"))

    # Detectar fallback
    fallback_markers = [
        "não pode ser encontrada",
        "não pode ser encontrado",
        "não foi encontrada",
        "não foi encontrado",
        "não há informações suficientes",
    ]

    is_fallback = any(marker in answer.lower() for marker in fallback_markers)

    # Se for fallback → Nao exibir fontes
    if is_fallback:
        return answer, source_docs

    # Caso contrário → formatar fontes normalmente
    formatted_sources = []

    for doc in source_docs[:5]:
        page = doc.metadata.get("page", "N/A")
        article = doc.metadata.get("article")

        if article:
            formatted_sources.append(f"{article} (p. {page})")
        else:
            formatted_sources.append(
                f"Página {page} (artigo não identificado no chunk)"
            )

    # Remove duplicados mantendo ordem
    formatted_sources = list(dict.fromkeys(formatted_sources))

    if formatted_sources:
        sources_text = "\n\nFontes:\n" + "\n".join(formatted_sources)
        final_answer = answer + sources_text
    else:
        final_answer = answer

    return final_answer, source_docs


def chat(chat_history, vectordb):
    user_query = st.chat_input("Faça uma pergunta:")
    if user_query:
        response, context = get_response(user_query, chat_history, vectordb)

        chat_history = chat_history + [
            HumanMessage(content=user_query),
            AIMessage(content=response),
        ]

    for message in chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
            st.write(message.content)

    return chat_history
