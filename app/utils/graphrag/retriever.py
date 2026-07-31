"""
Rtriever híbrido: busca vetorial para achar artigos-semente + expansão no
grafo para trazer artigos conectados que a similaridade semântica sozinha
não pega.

Isso ataca diretamente o gargalo identificado na V2(RAG) (45,2% das queries
in-scope com zero artigos recuperados): boa parte dessas perguntas depende
de mais de um artigo relacionado (ex.: pergunta no gold set que exige "arts.
1º e 3º" juntos), e busca por similaridade de texto não captura relação
jurídica entre artigos — só similaridade lexical/semântica de superfície.

Diferente da V1/V2 (que retornam chunks fragmentados), aqui o contexto
retornado ao LLM é sempre o artigo completo — outra correção direta de um
problema identificado no RAG (chunks diluídos, ver README).
"""

from dataclasses import dataclass, field

import networkx as nx

# Artigos com menos que isso de texto são, na prática, "(VETADO)" ou similares
# — sem conteúdo jurídico substantivo. Embeddings de texto muito curto tendem
# a ficar "genéricos" e artificialmente próximos de qualquer pergunta (esse
# problema já existia silenciosamente na V2: 'Art. 55' — inteiramente vetado —
# aparecia nos resultados de 30 das 36 perguntas do gold set). Filtrar aqui
# evita que esses artigos sirvam de âncora tanto pra busca vetorial quanto
# pra expansão no grafo.
MIN_ARTICLE_LENGTH = 100


def _is_substantive(G: nx.MultiDiGraph, article_id: str) -> bool:
    return len(G.nodes[article_id].get("text", "")) >= MIN_ARTICLE_LENGTH


@dataclass
class GraphRAGResult:
    context: str
    seed_articles: list[str]
    expanded_articles: list[str]
    all_articles: list[str] = field(init=False)

    def __post_init__(self):
        self.all_articles = self.seed_articles + self.expanded_articles


def _expand_via_graph(
    G: nx.MultiDiGraph,
    seed_articles: list[str],
    max_expanded: int = 5,
) -> list[str]:
    """
    a partir dos artigos-semente, encontra artigos relacionados por:
      (a) referência cruzada explícita (aresta REFERENCIA, nas duas direções)
      (b) entidade em comum (Artigo -> Entidade <- outro Artigo)

    Pntua candidatos pelo número de conexões distintas aos artigos-semente
    (mais conexões = mais provável de ser relevante) e retorna os top N,
    excluindo os próprios artigos-semente
    """
    seed_set = set(seed_articles)
    scores: dict[str, int] = {}

    for seed in seed_articles:
        if seed not in G:
            continue

        # (a) referências cruzadas diretas, nas duas direções
        for neighbor in list(G.successors(seed)) + list(G.predecessors(seed)):
            if G.nodes[neighbor].get("node_type") != "Artigo":
                continue
            if neighbor in seed_set or not _is_substantive(G, neighbor):
                continue
            scores[neighbor] = (
                scores.get(neighbor, 0) + 2
            )  # referência explícita pesa mais

        # (b) artigos que compartilham uma entidade com o artigo-semente
        for entity in G.successors(seed):
            if G.nodes[entity].get("node_type") == "Artigo":
                continue  # já tratado no passo (a)
            for other_article in G.predecessors(entity):
                if other_article in seed_set or other_article == seed:
                    continue
                if G.nodes[other_article].get("node_type") != "Artigo":
                    continue
                if not _is_substantive(G, other_article):
                    continue
                scores[other_article] = scores.get(other_article, 0) + 1

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [article_id for article_id, _ in ranked[:max_expanded]]


def _format_article_block(G: nx.MultiDiGraph, article_id: str, label: str) -> str:
    text = G.nodes[article_id].get("text", "")
    return f"[{label}: {article_id}]\n{text}"


def graph_retrieve(
    question: str,
    vectordb,
    G: nx.MultiDiGraph,
    k_seed_chunks: int = 15,
    max_seed_articles: int = 6,
    max_expanded_articles: int = 5,
) -> GraphRAGResult:
    """
    recuperação híbrida para uma pergunta.

    parametros:
        vectordb: o Chroma vectorstore já existente (chunks com metadata
            'article').
        G: grafo de conhecimento carregado (graph_builder.load_graph()).
    """
    # 1. busca vetorial para achar artigos-semente
    docs = vectordb.similarity_search(question, k=k_seed_chunks)

    seed_articles: list[str] = []
    for doc in docs:
        article = doc.metadata.get("article")
        if not article:
            continue
        article = _normalize_article_id(article)
        if article not in G or not _is_substantive(G, article):
            continue  # ignora artigos fora do grafo ou degenerados (ex.: "(VETADO)")
        if article not in seed_articles:
            seed_articles.append(article)
        if len(seed_articles) >= max_seed_articles:
            break

    # 2. Expansão via grafo
    expanded_articles = (
        _expand_via_graph(G, seed_articles, max_expanded=max_expanded_articles)
        if seed_articles
        else []
    )

    # 3. monta contexto (artigo completo, não chunk fragmentado)
    blocks = [
        _format_article_block(G, a, "recuperado por similaridade")
        for a in seed_articles
    ]
    blocks += [
        _format_article_block(G, a, "expandido via grafo") for a in expanded_articles
    ]

    return GraphRAGResult(
        context="\n\n---\n\n".join(blocks),
        seed_articles=seed_articles,
        expanded_articles=expanded_articles,
    )


def _normalize_article_id(raw: str) -> str:
    """normaliza ids de artigo vindos da metadata dos chunks (ex.: 'Art. 5º' -> 'Art. 5')."""
    import re

    m = re.match(r"Art\.?\s*(\d+)[ºo°]?(-([A-Z]))?", raw.strip())
    if not m:
        return raw
    num, _, suf = m.groups()
    return f"Art. {num}{'-' + suf if suf else ''}"
