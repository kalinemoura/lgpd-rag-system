"""
Diagnóstico: descobre qual artigo sumiu na segmentação.
Não chama a API, só roda o parser de novo e compara com a lista esperada.

rodar da raiz do projeto (rag_chatbot/):
    python check_articles.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

from pypdf import PdfReader
from utils.graphrag.parser import parse_articles

reader = PdfReader("docs/Texto LGPD.pdf")
full_text = "\n".join(page.extract_text() for page in reader.pages)

articles = parse_articles(full_text)

expected_plain = {f"Art. {i}" for i in range(1, 66)}
expected_suffixed = {f"Art. 55-{l}" for l in "ABCDEFGHIJKLM"} | {
    "Art. 58-A",
    "Art. 58-B",
}
expected = expected_plain | expected_suffixed

found = set(articles.keys())

missing = expected - found
extra = found - expected

print(f"Total encontrado: {len(found)} (esperado: {len(expected)})")

if missing:
    print(f"\nFALTANDO: {sorted(missing)}")
    for art_id in missing:
        # tenta achar o cabeçalho "quebrado" no texto bruto pra mostrar o contexto
        num = re.search(r"\d+", art_id).group()
        for m in re.finditer(re.escape(num), full_text):
            snippet = full_text[max(0, m.start() - 15) : m.start() + 25]
            if "Art" in snippet:
                print(f"  possível cabeçalho quebrado perto de '{art_id}': {snippet!r}")
else:
    print("\nNenhum artigo faltando — os 80 bateram certinho.")

if extra:
    print(f"\nIDs INESPERADOS (não deveriam existir): {sorted(extra)}")
