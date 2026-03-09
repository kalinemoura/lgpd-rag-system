import json
import pandas as pd
from rag_chatbot.app.utils.chatbot import get_response
from rag_chatbot.app.utils.prepare_vectordb import get_vectorstore
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

load_dotenv()

# =============================
# CONFIGURAÇÕES
# =============================

GOLDSET_PATH = "rag_chatbot/analysis/evaluation/gold_set_v1.json"
OUTPUT_PATH = "rag_chatbot/analysis/evaluation/execucao_rag_v2.xlsx"
PDFS = [r"C:\Users\User\Desktop\projeto-rag\rag_chatbot\docs\Texto LGPD.pdf"]

# =============================
# MODELO DE EMBEDDING (Cosine)
# =============================

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def calcular_cosine(resposta_modelo, resposta_esperada):
    emb_modelo = embedding_model.encode([resposta_modelo])
    emb_gold = embedding_model.encode([resposta_esperada])
    score = cosine_similarity(emb_modelo, emb_gold)[0][0]
    return float(score)


# =============================
# CARREGA VECTOR DB
# =============================

vectordb = get_vectorstore(PDFS, from_session_state=True)

# =============================
# CARREGA GOLDSET JSON
# =============================

with open(GOLDSET_PATH, "r", encoding="utf-8") as f:
    goldset = json.load(f)

df = pd.DataFrame(goldset)

results = []

# =============================
# LOOP PRINCIPAL
# =============================

for _, row in df.iterrows():
    pergunta = row["pergunta"]
    resposta_esperada = row["resposta_esperada"]

    print(f"\nRodando pergunta: {pergunta}")

    resposta_modelo, source_docs = get_response(
        question=pergunta,
        chat_history=[],
        vectordb=vectordb,
    )

    # =============================
    # MONTA CONTEXTO CONCATENADO
    # =============================

    contexto_texto = ""
    artigos = []

    for i, doc in enumerate(source_docs):
        article = doc.metadata.get("article", "Artigo não identificado")
        page = doc.metadata.get("page", "N/A")

        artigos.append(str(article))

        contexto_texto += (
            f"\n\n[Chunk {i + 1} - {article} - p.{page}]\n{doc.page_content}\n"
        )

    # =============================
    # CALCULA COSINE
    # =============================

    cosine_score = calcular_cosine(resposta_modelo, resposta_esperada)

    results.append(
        {
            "id": row.get("id"),
            "categoria": row.get("categoria"),
            "tipo_pergunta": row.get("tipo_pergunta"),
            "pergunta": pergunta,
            "resposta_esperada": resposta_esperada,
            "resposta_modelo": resposta_modelo,
            "cosine_score": cosine_score,
            "top_k_contexto": contexto_texto,
            "artigos_recuperados": ", ".join(list(dict.fromkeys(artigos))),
        }
    )

# =============================
# SALVA PLANILHA FINAL
# =============================

final_df = pd.DataFrame(results)
final_df.to_excel(OUTPUT_PATH, index=False)

print("\nExecução RAG gerada com sucesso")
