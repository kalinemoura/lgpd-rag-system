"""
Cnstrução do grafo de conhecimento (networkX) a partir das extrações por
artigo.

Modelo do grafo (equivalente em espírito à Figura 6.3 do livro Essential graphhrag, adaptado
para o domínio LGPD e sem banco externo):

    (Artigo) -[DEFINE|ESTABELECE|MENCIONA]-> (Entidade: Conceito/Direito/...)
    (Artigo) -[REFERENCIA]-> (Artigo)          # citações cruzadas explícitas

rsolução de entidade: simples, por normalização de string (tipo + nome em
minusculas, sem espaços extras). Suficiente para este domínio fechado ,o
livro (seção 6.2.2) já observa que resolução de entidade é específica de
domínio e que uma solução generica raramente funciona bem; aqui o LLM já é
instruído a usar nomes canônicos curtos, o que reduz bastante a necessidade
de resolução adicional
"""

import pickle
from pathlib import Path

import networkx as nx

from .parser import extract_cross_references
from .schema import ArticleExtraction, FIELD_TO_RELATION


def _normalize_entity_key(tipo: str, nome: str) -> str:
    return f"{tipo}:{nome.strip().lower()}"


def build_graph(
    articles: dict[str, str],
    extractions: dict[str, ArticleExtraction],
) -> nx.MultiDiGraph:
    """
    nonta o grafo completo a partir dos textos dos artigos e das extrações
    estruturadas correspondentes
    """
    G = nx.MultiDiGraph()

    # 1. nós de artigo, com o texto integral guardado no nó (é isso que o
    #    retriever vai usar como contexto, artigo inteiro, não fragmento).
    for article_id, text in articles.items():
        G.add_node(article_id, node_type="Artigo", text=text)

    # 2. entidades + arestas Artigo -> Entidade
    for article_id, extraction in extractions.items():
        if article_id not in G:
            continue  # segurança, não deveria acontecer

        for field_name, (relation, entity_type) in FIELD_TO_RELATION.items():
            for mention in getattr(extraction, field_name):
                entity_key = _normalize_entity_key(entity_type, mention.nome)

                if entity_key not in G:
                    G.add_node(
                        entity_key,
                        node_type=entity_type,
                        nome=mention.nome,
                        descricoes=[],
                    )
                # acumula descrições vindas de artigos diferentes
                G.nodes[entity_key]["descricoes"].append(
                    {"artigo": article_id, "descricao": mention.descricao}
                )

                G.add_edge(article_id, entity_key, relation=relation)

    # 3. referências cruzadas explicitas entre artigos (regex, barato e
    #    determinístico , complementa o que o LLM extraiu semanticamente)
    for article_id, text in articles.items():
        for ref_id in extract_cross_references(article_id, text):
            if ref_id in G:
                G.add_edge(article_id, ref_id, relation="REFERENCIA")

    return G


def save_graph(
    G: nx.MultiDiGraph, path: str = "graph_cache/lgpd_graph.gpickle"
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_graph(path: str = "graph_cache/lgpd_graph.gpickle") -> nx.MultiDiGraph:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Grafo não encontrado em '{path}'. Rode 'python -m utils.graphrag.build_graph' "
            "primeiro para gerar o cache (esse passo chama a API da OpenAI e não deve "
            "rodar a cada start do app)."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def graph_summary(G: nx.MultiDiGraph) -> str:
    """resumo rápido para conferencia manual após a construção."""
    node_types = {}
    for _, data in G.nodes(data=True):
        t = data.get("node_type", "?")
        node_types[t] = node_types.get(t, 0) + 1

    edge_types = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "?")
        edge_types[r] = edge_types.get(r, 0) + 1

    lines = [
        f"Nós totais: {G.number_of_nodes()}",
        f"Arestas totais: {G.number_of_edges()}",
        "",
    ]
    lines.append("Por tipo de nó:")
    for t, count in sorted(node_types.items(), key=lambda x: -x[1]):
        lines.append(f"  {t}: {count}")
    lines.append("\nPor tipo de relação:")
    for r, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        lines.append(f"  {r}: {count}")
    return "\n".join(lines)
