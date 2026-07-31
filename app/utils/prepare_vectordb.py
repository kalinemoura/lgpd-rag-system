from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import re


def infer_doc_type(filename: str) -> str:
    name = filename.lower()
    if "lgpd" in name:
        return "LGPD"
    return "Desconhecido"


ARTICLE_PATTERN = re.compile(r"Art\.?\s*\d+[º°]?(?:-[A-Z])?")


def extract_pdf_text(pdfs):
    """
    extrai o texto de cada PDF e já reagrupa por ARTIGO, não por página.

    por quê: com chunking por tamanho fixo (ex.: 2000 caracteres), um único
    chunk frequentemente contém VÁRIOS artigos inteiros (a LGPD tem muitos
    artigos curtos, ex. arts. 14 a 27). O metadata só conseguia guardar um
    artigo "primário" por chunk, então o conteúdo dos outros artigos que
    estavam no mesmo chunk ficava com o rótulo errado nas fontes, mesmo
    tendo sido usado pelo LLM pra responder.

    aqqui cada "documento" retornado já corresponde ao texto de UM único
    artigo (do "Art. N" até o próximo "Art. M", ou até o fim do texto).
    O chunking (get_text_chunks) roda depois, só para quebrar artigos muito
    longos em pedaços menores — e como cada pedaço vem de um artigo só,
    o metadata continua correto em qualquer sub-chunk, sem heurística
    nenhuma
    """
    article_docs = []

    for pdf in pdfs:
        pdf_path = os.path.join("docs", pdf)
        pages = PyPDFLoader(pdf_path).load()
        doc_type = infer_doc_type(pdf)

        # concatena todas as páginas do PDF num texto só, guardando em que
        # posição (offset) cada página começa — pra depois saber de qual
        # página cada artigo veio.
        full_text = ""
        page_offsets = []  # [(offset_no_texto_completo, numero_da_pagina), ...]
        for page in pages:
            page_offsets.append((len(full_text), page.metadata.get("page")))
            full_text += page.page_content + "\n"

        def page_for_offset(offset, _page_offsets=page_offsets):
            page_num = _page_offsets[0][1]
            for start, num in _page_offsets:
                if start <= offset:
                    page_num = num
                else:
                    break
            return page_num

        matches = list(ARTICLE_PATTERN.finditer(full_text))

        if not matches:
            # Nnhum "Art. N" encontrado no PDF inteiro (não deveria
            # acontecer com a LGPD, mas evita quebrar em outros documentos).
            article_docs.append(
                Document(
                    page_content=full_text,
                    metadata={
                        "filename": pdf,
                        "doc_type": doc_type,
                        "page": page_for_offset(0),
                    },
                )
            )
            continue

        # texto antes do primeiro "Art." (título da lei, ementa, preâmbulo)
        # vira um bloco sem artigo associado
        if matches[0].start() > 0:
            preamble = full_text[: matches[0].start()]
            if preamble.strip():
                article_docs.append(
                    Document(
                        page_content=preamble,
                        metadata={
                            "filename": pdf,
                            "doc_type": doc_type,
                            "page": page_for_offset(0),
                        },
                    )
                )

        # cada span vai do início de um "Art. N" até o início do próximo —
        # ou seja, o texto completo daquele artigo (incisos, parágrafos etc.)
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            article_text = full_text[start:end]

            article_docs.append(
                Document(
                    page_content=article_text,
                    metadata={
                        "filename": pdf,
                        "doc_type": doc_type,
                        "page": page_for_offset(start),
                        "article": match.group(0),
                    },
                )
            )

    return article_docs


def get_text_chunks(docs):
    """
    Qebra em pedaços menores SÓ quando um artigo é longo demais para caber
    num chunk. Como cada `doc` de entrada já pertence a um único artigo
    (ver extract_pdf_text), o metadata["article"] é simplesmente copiado
    para os sub-chunks pelo próprio langchain — e continua correto, sem
    precisar de nenhuma lógica extra de inferência

    Parameters:
    - docs (list): Lista de documentos, cada um já correspondendo a um
      único artigo (ou ao preâmbulo, sem artigo)

    Returns:
    - chunks: List de chunks de texto
    """
    # chunk_size reduzido de 2000 para 600. O valor anterior (2000 chars)
    # tentava aproximar um limite de tokens do modelo, mas o cálculo estava
    # errado (2000 chars ~ 400-500 tokens, não 2048) — e na prática deixava
    # artigos com muitos incisos (ex. Art. 18, com 9 incisos + parágrafos)
    # inteiros num chunk só. Isso "dilui" o embedding: o vetor do chunk vira
    # uma média de vários temas diferentes, o que piora o ranking de
    # similaridade até para perguntas bem específicas sobre um inciso só.
    # Com chunk_size menor, artigos longos são divididos em pedaços mais
    # focados (2-3 incisos por vez), mantendo o metadata "article" correto
    # em cada pedaço (herdado do documento de origem, ver extract_pdf_text).
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def get_vectorstore(pdfs, from_session_state=False):
    load_dotenv()

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Caso 1: carregar DB existente
    if from_session_state and os.path.exists("Vector_DB - Documents"):
        return Chroma(
            persist_directory="Vector_DB - Documents", embedding_function=embedding
        )

    # caso 2: criar DB do zero
    docs = extract_pdf_text(pdfs)
    chunks = get_text_chunks(docs)

    print("\n=== DEBUG METADATA (primeiros 3 chunks) ===")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i}")
        print("Metadata:", chunk.metadata)
        print("Preview:", chunk.page_content[:200])

    if not chunks:
        raise ValueError("No text chunks could be created from the PDF.")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="Vector_DB - Documents",
    )
