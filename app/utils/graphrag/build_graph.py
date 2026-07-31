"""
Script para construir o grafo de conhecimento da LGPD offline, uma única vez.

Isso chama a API da OpenAI (uma vez por artigo, ~80 chamadas com gpt-4o-mini
,rápido e barato) e salva o grafo resultante em disco. O app do Streamlit
só carrega o grafo já pronto (utils/graphrag/graph_builder.load_graph), não
refaz a extração a cada execução

Os caminhos (PDF de entrada e cache de saída) são resolvidos a partir da
localização desse arquivo, ñ do diretório de onde  roda o comando,
então funciona tanto rodando da raiz do projeto (rag_chatbot/) quanto de
dentro de app/.

Rodar da raiz do projeto (mesmo lugar de onde roda `streamlit run app/app.py`):

    python app/utils/graphrag/build_graph.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

# app/utils/graphrag/build_graph.py -> sobe 3 níveis até a raiz do projeto
_THIS_FILE = Path(__file__).resolve()
ROOT_DIR = _THIS_FILE.parents[3]
APP_DIR = _THIS_FILE.parents[2]

# permite importar "utils.graphrag..." independente de como o script foi chamado
sys.path.insert(0, str(APP_DIR))

from utils.graphrag.parser import parse_articles
from utils.graphrag.extraction import extract_all_articles
from utils.graphrag.graph_builder import build_graph, save_graph, graph_summary

PDF_PATH = ROOT_DIR / "docs" / "Texto LGPD.pdf"
GRAPH_CACHE_PATH = ROOT_DIR / "graph_cache" / "lgpd_graph.gpickle"


def main():
    load_dotenv(ROOT_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY não encontrada. Configure no .env (na raiz do projeto) "
            "antes de rodar este script."
        )

    print(f"Lendo {PDF_PATH}...")
    reader = PdfReader(str(PDF_PATH))
    full_text = "\n".join(page.extract_text() for page in reader.pages)

    print("Segmentando por artigo...")
    articles = parse_articles(full_text)
    print(f"  {len(articles)} artigos identificados.\n")

    print("Extraindo entidades via LLM (isso consome créditos de API)...")
    extractions = extract_all_articles(articles)

    print("\nConstruindo grafo...")
    G = build_graph(articles, extractions)

    save_graph(G, str(GRAPH_CACHE_PATH))
    print(f"\nGrafo salvo em '{GRAPH_CACHE_PATH}'.\n")
    print(graph_summary(G))


if __name__ == "__main__":
    main()
