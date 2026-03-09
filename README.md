# Consultor LGPD – RAG Jurídico

Este projeto implementa um chatbot baseado em RAG (Retrieval-Augmented Generation) especializado na Lei Geral de Proteção de Dados (Lei nº 13.709/2018).

O sistema responde com base em trechos recuperados da legislação, buscando reduzir alucinações e fornecer rastreabilidade das respostas por meio da citação explícita das fontes (artigo e página).

## Objetivo

Explorar práticas de construção de sistemas RAG confiáveis aplicados a textos legais, com foco em:

- Redução de respostas inventadas (alucinação)
- Rastreabilidade das respostas por meio de citações
- Uso de recuperação semântica para fundamentar as respostas
- Base para avaliações sistemáticas futuras

> **Nota 1:** Este projeto foi desenvolvido a partir do repositório [vitorccmanso/Rag-ChatBot](https://github.com/vitorccmanso/Rag-ChatBot), com adaptações para execução local e uso de modelos de embedding com pesos públicos,
> executáveis localmente.

> **Nota 2:** Nesta fase, optou-se por manter o texto integral dos documentos, incluindo notas editoriais e trechos revogados, a fim de estabelecer um baseline realista para avaliação do sistema. A limpeza e normalização do texto são consideradas como trabalhos futuros.

## Interface

![RAG LGPD Chatbot](Images/app.interface.png)

## Funcionalidades

- Ingestão e processamento automático de PDF jurídico (base fixa – LGPD)
- Geração de embeddings e indexação vetorial persistente
- Busca semântica (retrieval top-k)
- Respostas geradas exclusivamente com base no contexto recuperado
- Exibição de citações (documento, artigo e página)
- Controle de fallback para evitar alucinações

## Arquitetura

Pipeline do sistema:

1. Ingestão do PDF da LGPD.
2. Chunking com `RecursiveCharacterTextSplitter` 
3. Geração de embeddings (`all-MiniLM-L6-v2`)
4. Armazenamento vetorial com ChromaDB
5. Retrieval semântico (top-k)
6. Geração de resposta via LLM
7. Exibição de citações (artigo + página)
8. Fallback controlado quando não há contexto suficiente

## Decisões Técnicas

### Chunking
- **Estratégia:** `RecursiveCharacterTextSplitter`
- **Parâmetros:**
  - `chunk_size`: 2000 caracteres
  - `chunk_overlap`: 300 caracteres
  - Baseado na análise estatística dos artigos da LGPD
  - Aproximadamente 86% dos artigos permanecem íntegros em um único chunk
- **Objetivo:** Preservar contexto jurídico

### Embeddings
- **Modelo:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vantagens:**
  - Execução local (sem custos de API)
  - Boa performance para recuperação semântica e adequado para v1
  - Leve e eficiente

### Retrieval
- `top_k = 10` para priorizar recall em contexto jurídico, onde a resposta pode estar distribuída em múltiplos dispositivos legais
- Re-ranking planejado para versão futura

### Modelo de Linguagem (LLM)
- **Modelo:** OpenAI GPT-4o-mini
- **Temperatura:** 0.2 (para reduzir variabilidade)
- **Prompt:** Instruções restritivas para responder apenas com base no contexto recuperado

### Vector Store
- **Banco:** ChromaDB
- **Persistência:** Local em disco
- **Benefício:** Reuso automático do índice entre execuções

### Camada de Confiabilidade
Exibição de:
- Documento
- Artigo
- Página
- Formatação amigável para o usuário

## Como Executar

### Pré-requisitos

- Python 3.10 ou superior
- Chave de API da OpenAI

### Instalação

1. Clone o repositório:
```bash
git clone <seu-repositorio>
cd <nome-do-projeto>
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
├── app/
│   └── app.py             
├── requirements.txt        
├── .env                    
└── README.md              
```

## Limitações Conhecidas

- O fallback não solicita reformulação da pergunta; optou-se por resposta direta para priorizar controle de alucinação na v1
- Similaridade semântica pode não ranquear corretamente artigos definidores (ex: Art. 5º).
- Chunk pode não herdar metadata quando começa em inciso.
- Perguntas amplas podem acionar fallback.
- Não há re-ranking.
- Não há query rewriting.
- Não há avaliação automática estruturada.


## Próximos Passos 

### Versão 2 — Avaliação e Diagnóstico

Introdução de um processo estruturado de avaliação para analisar a qualidade das respostas.

Inclui:
- construção de um conjunto de perguntas de teste (gold set)
- métricas quantitativas de qualidade
- análise sistemática de padrões de erro
- diagnóstico do comportamento do retrieval

**Objetivo:** Identificar limitações empiricamente e priorizar melhorias baseadas em dados

### Versão 3 — Interpretação Semântica de Perguntas

Camada de NLU para interpretação estruturada de consultas jurídicas.

Possíveis direções:
- classificação de intenção da pergunta
- extração de entidades normativas
- normalização de termos jurídicos

**Objetivo:** Melhorar alinhamento entre consulta e contexto recuperado

**Desenvolvido como projeto de estudo em RAG confiável**

