"""
Extração de entidades por artigo via OpenAI structured outputs.

Segue o padrão do Capítulo 6 de "Essential GraphRAG": chamada direta à API
da OpenAI com response_format=<modelo Pydantic>, em vez de passar pelo
LangChain (mantém a extração simples e isolada do resto do pipeline RAG,
que continua em LangChain).
"""

import time
from openai import OpenAI, RateLimitError

from .schema import ArticleExtraction, SYSTEM_MESSAGE

DEFAULT_MODEL = "gpt-4o-mini"


def extract_article(
    article_id: str,
    article_text: str,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
) -> ArticleExtraction:
    """
    extrai entidades estruturadas de um único artigo da LGPD.

    parametros:
        article_id: identificador do artigo (ex.: "Art. 5"), usado só para
            contexto no prompt e em mensagens de erro.
        article_text: texto integral do artigo.
        client: cliente OpenAI já instanciado (reutilize entre chamadas para
            não recriar conexão a cada artigo). Se None, cria um novo.
        model: modelo com suporte a Structured outputs

    returns:
        ArticleExtraction preenchido. Em caso de falha após retries, retorna
        uma extração vazia (todas as listas vazias) pra não interromper o
        processamnto dos demais artigos
    """
    client = client or OpenAI()

    user_message = f"{article_id}\n\n{article_text}"

    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_message},
                ],
                response_format=ArticleExtraction,
            )
            return response.choices[0].message.parsed
        except RateLimitError:
            wait = 2**attempt
            print(
                f"  [rate limit] aguardando {wait}s e tentando de novo ({article_id})..."
            )
            time.sleep(wait)
        except Exception as e:
            print(
                f"  [erro] falha ao extrair {article_id} (tentativa {attempt + 1}): {e}"
            )
            time.sleep(1)

    print(f"  [aviso] {article_id} ficou sem extração após {max_retries} tentativas.")
    return ArticleExtraction()


def extract_all_articles(
    articles: dict[str, str],
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> dict[str, ArticleExtraction]:
    """
    Roda a extração para todos os artigos. Este é o passo que consome tokens/
    tempo de API — pensado para rodar uma vez offline (ver build_graph.py) e
    ter o resultado cacheado em disco, não a cada execução do app
    """
    client = OpenAI()
    results = {}
    total = len(articles)

    for i, (article_id, text) in enumerate(articles.items(), 1):
        if verbose:
            print(f"[{i}/{total}] Extraindo {article_id}...")
        results[article_id] = extract_article(
            article_id, text, client=client, model=model
        )

    return results
