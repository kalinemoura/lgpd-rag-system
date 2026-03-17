# Consultor LGPD – RAG Jurídico (V2: Avaliação e Diagnóstico)

Esta versão avalia sistematicamente o desempenho do sistema RAG desenvolvido 
na V1, com foco em identificar limitações, medir qualidade objetivamente, e 
diagnosticar o comportamento do retrieval.


## Objetivo

Diagnosticar empiricamente onde e por que o sistema falha, estabelecendo 
baseline quantitativo para orientar melhorias futuras.


## Metodologia 

A avaliação foi conduzida utilizando um gold set de 36 perguntas anotadas 
manualmente, cobrindo diferentes categorias:

- conceitos legais
- bases legais
- direitos do titular
- obrigações do controlador
- sanções administrativas
- perguntas ambíguas
- perguntas incompletas
- perguntas fora de escopo

Para cada consulta foram registrados:

- resposta gerada pelo sistema
- artigos legais esperados (ground truth)
- trechos (chunks) recuperados pelo mecanismo de busca
- avaliação manual da qualidade da resposta
- métricas automáticas de avaliação
- classificação do comportamento do sistema (resposta direta ou fallback)


## Sistema Avaliado

A avaliação foi realizada sobre o pipeline desenvolvido na Versão 1, composto por:

- Ingestão do PDF da LGPD
- Chunking com `RecursiveCharacterTextSplitter`
- Embeddings `all-MiniLM-L6-v2`
- Indexação vetorial com ChromaDB
- Retrieval semântico (top-k = 10)
- Geração via GPT-4o-mini com prompt restritivo
- Mecanismo de fallback controlado


## Avaliação do Retrieval

Considerando apenas perguntas in-scope (com artigo esperado):

**Zero artigos recuperados:** 45,2% (Crítico)
- Quase metade das queries falha completamente

**Pelo menos 1 artigo:** 54,8% (Moderado)
- Maioria recupera algum contexto relevante

**Todos artigos esperados:** 51,6% (Moderado)
- Metade consegue cobertura completa

**Padrão:** Comportamento polarizado (tudo ou nada) - quando acerta, acerta bem; quando erra, erra completamente.

**Conclusão:** Retrieval é o principal gargalo do sistema.


## Avaliação Completa

A avaliação detalhada do sistema foi realizada por meio de um notebook dedicado, que documenta a metodologia, as métricas utilizadas e a análise dos resultados.

Notebook:

[analyse_results_v2.ipynb](analysis/evaluation/analyse_results_v2.ipynb)

Dados utilizados na avaliação (gold set e anotações):

[avaliacao_v2_final.xlsx](analysis/evaluation/avaliacao_v2_final.xlsx)


## Resultados e Observações

### Pontos Fortes Identificados
- Sistema reconhece adequadamente perguntas fora de escopo
- Zero alucinações detectadas (todas respostas baseadas em contexto)
- LLM-as-judge teve melhor alinhamento com avaliação humana que similaridade semântica

### Limitações Identificadas
- **Gargalo crítico no retrieval:** 45,2% das queries não recuperam artigo esperado
- Fallback acionado com frequência excessiva e nem sempre apropriado
- Fallback não sugere reformulação (limita UX)
- Muitas respostas incorretas decorrem de contexto inadequado/insuficiente

### Insights Técnicos
- Retrieval tem comportamento polarizado (tudo ou nada)
- Rotulagem inconsistente de chunks dificulta análise precisa
- Similaridade semântica não garante qualidade jurídica


## Limitações da Avaliação

- Em alguns casos, inconsistências na rotulagem automática dos chunks dificultaram a identificação precisa do artigo recuperado.
- Foi utilizada uma verificação manual (artigo_real_recuperado) para mitigar esse problema.
- Pequenas imprecisões residuais podem afetar marginalmente as métricas de retrieval.


## Estrutura da Avaliação

```
analysis/
└── evaluation/
    ├── analyse_results_v2.ipynb
    └── avaliacao_v2_final.xlsx

```

# Próximos Passos

## Versão 3 — Otimização do Retrieval

Com base nos resultados obtidos, a próxima etapa do projeto foca na melhoria do mecanismo de recuperação de informações.

Principais direções:

- melhorias no mecanismo de retrieval
- aprimoramento do mecanismo de fallback, incluindo sugestões de reformulação de perguntas
- melhoria na rotulagem dos chunks
- embeddings mais adequados ao domínio jurídico
- query rewriting
- reranking dos resultados
- possíveis abordagens híbridas de busca
- limpeza e normalização do texto da base normativa (remoção de trechos revogados e notas editoriais)

**Objetivo:** Aumentar a taxa de recuperação de pelo menos 1 artigo relevante 
de 54,8% para >80%, reduzindo falhas críticas (zero artigos) de 45,2% para <20%.






