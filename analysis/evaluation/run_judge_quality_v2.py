import json
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# =============================
# CONFIG
# =============================

INPUT_PATH = "rag_chatbot/analysis/evaluation/execucao_rag_v2.xlsx"
OUTPUT_PATH = "rag_chatbot/analysis/evaluation/execucao_rag_v2_com_judge1.xlsx"

llm_judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =============================
# LOAD BASE
# =============================

df = pd.read_excel(INPUT_PATH)

results = []

# =============================
# LOOP JUDGE (QUALIDADE JURÍDICA)
# =============================

for _, row in df.iterrows():
    pergunta = row["pergunta"]
    resposta_esperada = row["resposta_esperada"]
    resposta_modelo = row["resposta_modelo"]

    print(f"Julgando pergunta: {pergunta}")

    prompt = f"""
Você é um avaliador jurídico especializado em LGPD (Lei Geral de Proteção de Dados).

Sua tarefa é avaliar a qualidade jurídica da resposta do modelo comparando-a com a resposta esperada (gabarito).

Você NÃO deve considerar o contexto recuperado pelo RAG.
Avalie apenas a aderência ao gabarito.

---

## Pergunta:
{pergunta}

## Resposta Esperada (gabarito):
{resposta_esperada}

## Resposta do Modelo:
{resposta_modelo}

---

## Critérios (0 ou 1):

1. **correcao_factual**
   - 1 = sem erros factuais relevantes
   - 0 = contém erros ou contradições com o gabarito

2. **completude_juridica**
   - 1 = cobre os pontos essenciais do gabarito
   - 0 = omite elementos jurídicos importantes

---

## Nota Final (0 a 10):

- 2 critérios = 10
- 1 critério = 5
- 0 critérios = 0

---

## Classificação:

- "correta" se nota = 10
- "parcial" se nota = 5
- "incorreta" se nota = 0

---

Retorne APENAS um JSON válido no seguinte formato:

{{
  "correcao_factual": 0 ou 1,
  "completude_juridica": 0 ou 1,
  "nota_final": número,
  "classificacao": "correta|parcial|incorreta",
  "justificativa": "texto curto explicando a avaliação"
}}
"""

    response = llm_judge.invoke([HumanMessage(content=prompt)])

    try:
        parsed = json.loads(response.content)
    except:
        parsed = {
            "correcao_factual": None,
            "completude_juridica": None,
            "nota_final": None,
            "classificacao": "erro_parse",
            "justificativa_qualidade": response.content,
        }

    results.append(parsed)

judge_df = pd.DataFrame(results)

final_df = pd.concat([df, judge_df], axis=1)

final_df.to_excel(OUTPUT_PATH, index=False)

print("Judge 1 (Qualidade Jurídica) concluído com sucesso.")
