# Consultor LGPD – RAG Jurídico

Este projeto implementa um chatbot baseado em RAG (Retrieval-Augmented Generation) especializado na Lei Geral de Proteção de Dados (Lei nº 13.709/2018).

O sistema responde com base em trechos recuperados da legislação, buscando reduzir alucinações e fornecer rastreabilidade das respostas por meio da citação explícita das fontes (artigo e página).


## Objetivo do Projeto

Investigar e desenvolver um sistema RAG confiável aplicado a textos legais, evoluindo progressivamente por meio de versões experimentais e avaliação empírica.

- Redução de alucinações
- Rastreabilidade via citações explícitas
- Recuperação semântica fundamentada
- Avaliação rigorosa e melhorias baseadas em dados
- Identificação sistemática de limitações

> **Nota 1:** Este projeto foi desenvolvido a partir do repositório [vitorccmanso/Rag-ChatBot](https://github.com/vitorccmanso/Rag-ChatBot), com adaptações para execução local e uso de modelos de embedding com pesos públicos,
> executáveis localmente.

> **Nota 2:** Nesta fase, optou-se por manter o texto integral dos documentos, incluindo notas editoriais e trechos revogados, a fim de estabelecer um baseline realista para avaliação do sistema. A limpeza e normalização do texto são consideradas como trabalhos futuros.


## Roadmap do Projeto

### Versão 1 — Implementação (Concluída)

Sistema RAG jurídico funcional com pipeline completo.

**Principais features:**
- Chunking otimizado para textos jurídicos (86% dos artigos preservados íntegros)
- Embeddings locais (`all-MiniLM-L6-v2`)
- ChromaDB para indexação vetorial
- Retrieval semântico (top_k = 10)
- Geração via GPT-4o-mini com prompt restritivo
- Citação automática (artigo + página)
- Fallback controlado

**Limitações conhecidas:**
- Fallback não solicita reformulação
- Similaridade semântica pode falhar em artigos definidores
- Chunk pode não herdar metadata em alguns casos
- Sem re-ranking ou query rewriting

### Versão 2 — Avaliação e Diagnóstico (Concluída)

Avaliação empírica do sistema da V1 utilizando um gold set de 36 perguntas anotadas manualmente.

#### Metodologia

Para cada consulta foram registrados:
- Resposta gerada pelo sistema
- Artigos legais esperados
- Trechos recuperados pelo mecanismo de busca
- Avaliação manual da qualidade
- Métricas automáticas
- Classificação do comportamento (resposta direta ou fallback)

#### Resultados - Avaliação do Retrieval

Considerando apenas perguntas in-scope (com artigo esperado):

**Zero artigos recuperados:** 45,2% (Crítico)
- Quase metade das queries falha completamente

**Pelo menos 1 artigo:** 54,8% (Moderado)
- Maioria recupera algum contexto relevante

**Todos artigos esperados:** 51,6% (Moderado)
- Metade consegue cobertura completa

**Padrão identificado:** Comportamento polarizado (tudo ou nada) - quando acerta, acerta bem; quando erra, quando erra, falha completamente.

**Conclusão:** Retrieval é o principal gargalo do sistema.

#### Pontos Fortes Identificados

- Sistema reconhece adequadamente perguntas fora de escopo
- Zero alucinações detectadas (todas respostas baseadas em contexto)
- LLM-as-judge teve melhor alinhamento com avaliação humana que similaridade semântica

#### Limitações Identificadas

- Alta taxa de falha na recuperação de contexto
- Fallback acionado com frequência excessiva
- Ausência de sugestões de reformulação
- Muitas respostas incorretas decorrem de contexto inadequado/insuficiente

#### Análise detalhada disponível em:

- [Notebook de análise](./analysis/evaluation/analyse_results_v2.ipynb)
- [Dados da avaliação](./analysis/evaluation/avaliacao_v2_final.xlsx)


## Arquitetura

Pipeline do sistema:

1. Ingestão offline do texto da LGPD (base normativa fixa).
2. Chunking com `RecursiveCharacterTextSplitter` 
3. Geração de embeddings (`all-MiniLM-L6-v2`)
4. Armazenamento vetorial com ChromaDB
5. Retrieval semântico (top-k)
6. Geração de resposta via LLM
7. Exibição de citações (artigo + página)
8. Fallback controlado quando não há contexto suficiente


## Funcionalidades

- Perguntas em linguagem natural sobre LGPD
- Respostas fundamentadas exclusivamente no texto legal
- Citações explícitas das fontes (artigo e página)
- Detecção automática de perguntas fora de escopo
- Controle de alucinação via fallback
- Interface de chat simples para consulta jurídica


## Como Executar (V1)

### Pré-requisitos

- Python 3.10 ou superior
- Chave de API da OpenAI

### Instalação

1. Clone o repositório:
```bash
git clone 
cd 
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure a chave da API da OpenAI:

Crie um arquivo `.env` na raiz do projeto:
```env
OPENAI_API_KEY=sua_chave_aqui
```

### Execução

Execute o aplicativo Streamlit:
```bash
streamlit run app/app.py
```
O aplicativo estará disponível em `http://localhost:8501`


## Estrutura do Projeto
```
.
├── app/                    # V1: Aplicação Streamlit
│   └── app.py
├── analysis/               # V2: Avaliação
│   └── evaluation/
│       ├── analyse_results_v2.ipynb
│       └── avaliacao_v2_final.xlsx
├── requirements.txt
├── .env.example
└── README.md              # Este arquivo
```
