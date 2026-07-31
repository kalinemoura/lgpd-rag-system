from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from utils.graphrag.retriever import graph_retrieve

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Você é um assistente especializado na Lei Geral de Proteção de Dados (LGPD) que "
        "responde perguntas com base EXCLUSIVA no contexto fornecido.\n"
        "Use apenas as informações do contexto. Não utilize conhecimento externo.\n"
        "Não invente fatos ou definições.\n"
        "Responda sempre em português do Brasil.\n\n"
        "O contexto abaixo contém artigos recuperados por similaridade semântica com a "
        "pergunta, e artigos adicionais trazidos por conexão jurídica no grafo de "
        "conhecimento (referência cruzada explícita ou conceito/direito/base legal em "
        "comum), use ambos os tipos quando relevante para responder de forma completa.\n\n"
        "Se a resposta exigir combinar informação de mais de um artigo do contexto (ex.: um "
        "artigo estabelece um direito e outro estabelece uma exceção ou obrigação relacionada), "
        "sintetize a resposta considerando todos os artigos relevantes, em vez de responder que "
        "a informação não foi localizada.\n\n"
        "Se a resposta estiver claramente no contexto, responda de forma objetiva, citando "
        "os artigos utilizados.\n"
        "Se houver apenas informação parcial, responda apenas com o que estiver sustentado "
        "no texto.\n"
        "Só diga que a informação não foi localizada se o contexto fornecido realmente não "
        "tiver nenhum artigo relacionado ao tema da pergunta.\n\n"
        "IMPORTANTE — NÃO USE CONHECIMENTO PRÉVIO SOBRE A LGPD: você pode já conhecer o "
        "texto da LGPD de seu treinamento, mas para esta tarefa isso é proibido. Só cite um "
        "número de artigo (ex.: 'art. 7º') se o texto completo desse artigo estiver "
        "literalmente presente no bloco de Contexto abaixo. Nunca cite, descreva incisos "
        "de, ou faça referência a um artigo que não apareça no Contexto, mesmo que você "
        "'saiba' o que ele diz, isso conta como invenção de fato.\n\n"
        "Contexto:\n{context}\n\n"
        "Pergunta:\n{question}\n\n"
        "Resposta:"
    ),
)


def get_response_graphrag(question: str, vectordb, graph):
    """
    gerra uma resposta usando o pipeline GraphRAG: recuperação híbrida
    (vetorial + expansão no grafo) seguida de geração com o mesmo modelo/
    estilo de prompt da V2, para permitir comparação direta entre as duas
    versões na mema interface
    """
    load_dotenv()

    result = graph_retrieve(question, vectordb, graph)

    # debug - inspeção da recuperação híbrida
    print("\n=== DEBUG GRAPHRAG RETRIEVAL ===")
    print("Artigos-semente (busca vetorial):", result.seed_articles)
    print("Artigos expandidos (grafo):", result.expanded_articles)
    print(f"Tamanho do contexto montado: {len(result.context)} caracteres")
    print("--- Preview do contexto (primeiros 500 chars) ---")
    print(result.context[:500])
    print("=== FIM DEBUG ===\n")

    if not result.all_articles:
        return (
            "Não foi possível localizar artigos relevantes da LGPD para essa pergunta.",
            result,
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = PROMPT | llm

    response = chain.invoke({"context": result.context, "question": question})
    answer = response.content

    sources = []
    for article in result.seed_articles:
        sources.append(f"{article} (busca por similaridade)")
    for article in result.expanded_articles:
        sources.append(f"{article} (expandido via grafo)")

    final_answer = answer
    if sources:
        final_answer += "\n\nFontes:\n" + "\n".join(sources)

    return final_answer, result
