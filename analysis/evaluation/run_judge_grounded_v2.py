import json
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# =============================
# CONFIG
# =============================

INPUT_PATH = "rag_chatbot/analysis/evaluation/execucao_rag_v2_com_judge1.xlsx"
# OUTPUT_PATH = "rag_chatbot/analysis/evaluation/avaliacao_v2_final.xlsx"
OUTPUT_PATH = "rag_chatbot/analysis/evaluation/avaliacao_v2_grounded_corrigido.xlsx"

llm_judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =============================
# LOAD BASE
# =============================

df = pd.read_excel(INPUT_PATH)

results = []

# =============================
# LOOP GROUNDED
# =============================

for _, row in df.iterrows():
    pergunta = row["pergunta"]
    contexto = row["top_k_contexto"]
    resposta_modelo = row["resposta_modelo"]

    print(f"Avaliando groundedness: {pergunta}")

    prompt = f"""
Você é um avaliador jurídico especializado em LGPD  (Lei Geral de Proteção de Dados).

Sua ÚNICA tarefa é avaliar se a resposta está fundamentada EXCLUSIVAMENTE
no contexto fornecido.

Definição de groundedness:
- "fundamentado": todas as afirmações da resposta podem ser verificadas no contexto recuperado.
- "parcial": parte da resposta pode ser verificada no contexto, mas há afirmações adicionais não verificáveis.
- "nao_fundamentado": a resposta contém afirmações que não podem ser verificadas no contexto ou contradizem o contexto.

Regras obrigatórias:
- NÃO use conhecimento jurídico externo.
- Avalie apenas se as afirmações da resposta podem ser verificadas no contexto.
- Se a resposta apenas declarar que a informação não foi localizada nos trechos recuperados, 
  e não adicionar conteúdo externo, classifique como "fundamentado".
  
---

## Contexto Recuperado:
{contexto}

## Pergunta:
{pergunta}

## Resposta do Modelo:
{resposta_modelo}

Retorne APENAS um JSON válido:

{{
  "groundedness": "fundamentado|parcial|nao_fundamentado",
  "justificativa_grounded": "breve explicação objetiva"
}}
"""

    response = llm_judge.invoke([HumanMessage(content=prompt)])

    try:
        parsed = json.loads(response.content)
    except:
        parsed = {
            "groundedness": "erro_parse",
            "justificativa_grounded": response.content,
        }
    results.append(parsed)


grounded_df = pd.DataFrame(results)

# Remove colunas antigas se existirem
df = df.drop(columns=["groundedness", "justificativa_grounded"], errors="ignore")

final_df = pd.concat([df, grounded_df], axis=1)

final_df.to_excel(OUTPUT_PATH, index=False)
