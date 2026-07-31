"""
Parser de artigos da LGPD.

O texto oficial "compilado" da LGPD repete o cabeçalho de um mesmo artigo
sempre que existe uma redação histórica diferente (ex.: "(Redação dada pela
Lei nº 13.853, de 2019)"). Este parser agrupa todos os segmentos de um mesmo
número de artigo em um único texto, preservando o conteúdo integral (decisão
já registrada no README: limpeza/normalização fica para trabalho futuro).

Uso:
    from utils.graphrag.parser import parse_articles
    articles = parse_articles(texto_completo_lgpd)
    # articles: Dict[str, str]  ->  {"Art. 5": "Art. 5º Para os fins desta Lei...", ...}
"""

import re
from collections import defaultdict
from typing import Dict

# Casa "Art. 1º", "Art. 10.", "Art. 55-A." etc., apenas no início de linha
# (evita capturar citações soltas no meio do texto, tipo "nos termos do art. 7º").
HEADER_RE = re.compile(r"^Art\.\s*(\d+)[ºo°]?(-([A-Z]))?\.?\s", re.MULTILINE)

# O pypdf, ao extrair texto de docs/Texto LGPD.pdf, ocasionalmente insere um
# espaço espúrio dentro do número do artigo (ex.: "Art. 5 7." em vez de
# "Art. 57.") ou entre o número e o sufixo de letra (ex.: "55 -D" em vez de
# "55-D"). Sem essa correção, o Art. 57 fica grudado no Art. 5, e o Art. 55-D
# desaparece inteiramente (fica grudado no Art. 55-C). Corrigido aqui antes
# da segmentação.
_STRAY_DIGIT_SPACE_RE = re.compile(r"(Art\.\s*)(\d)\s(\d)([ºo°]?\.)")
_STRAY_HYPHEN_SPACE_RE = re.compile(r"(\d)\s+-([A-Z]\.)")


def _normalize_pdf_quirks(text: str) -> str:
    text = _STRAY_DIGIT_SPACE_RE.sub(r"\1\2\3\4", text)
    text = _STRAY_HYPHEN_SPACE_RE.sub(r"\1-\2", text)
    return text


# Casa citações de artigo em qualquer lugar do texto (para extrair referências
# cruzadas). Ex.: "art. 7º", "arts. 1º e 3º", "art. 11".
CITATION_RE = re.compile(r"art(?:igo)?s?\.?\s*(\d+)[ºo°]?(-([A-Z]))?", re.IGNORECASE)


def _normalize_id(num: str, suf: str | None) -> str:
    return f"Art. {num}{'-' + suf if suf else ''}"


def parse_articles(full_text: str) -> Dict[str, str]:
    """
    Divide o texto integral da LGPD em segmentos por artigo, concatenando
    redações históricas duplicadas sob o mesmo id.

    Returns:
        Dict article_id -> texto completo do artigo (todas as versões concatenadas)
    """
    full_text = _normalize_pdf_quirks(full_text)

    matches = list(HEADER_RE.finditer(full_text))
    if not matches:
        raise ValueError(
            "Nenhum cabeçalho de artigo encontrado. Verifique se o texto de entrada "
            "é o texto compilado da LGPD (extraído do PDF)."
        )

    grouped = defaultdict(list)
    for i, m in enumerate(matches):
        art_id = _normalize_id(m.group(1), m.group(3))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        grouped[art_id].append(full_text[start:end].strip())

    return {art_id: "\n\n".join(segments) for art_id, segments in grouped.items()}


def extract_cross_references(article_id: str, article_text: str) -> list[str]:
    """
    extrai, via regex, citaçoes explícitas a outros artigos dentro do texto de
    um artigo (ex.: "observado o disposto no art. 7º"). commplementa a extração
    por LLM com um sinal barato e deterministico de conexão entre artigos.

    exclui autocitação (um artigo citando a si mesmo).
    """
    refs = set()
    for m in CITATION_RE.finditer(article_text):
        candidate = _normalize_id(m.group(1), m.group(3))
        if candidate != article_id:
            refs.add(candidate)
    return sorted(refs)


if __name__ == "__main__":
    # smoke test manual: python -m utils.graphrag.parser docs/"Texto LGPD.pdf"
    import sys
    from pypdf import PdfReader

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "docs/Texto LGPD.pdf"
    reader = PdfReader(pdf_path)
    text = "\n".join(p.extract_text() for p in reader.pages)

    articles = parse_articles(text)
    print(f"{len(articles)} artigos identificados.\n")
    for art_id in list(articles)[:3]:
        print(f"--- {art_id} ---")
        print(articles[art_id][:200], "...\n")
        print(
            "Referências cruzadas:", extract_cross_references(art_id, articles[art_id])
        )
        print()
