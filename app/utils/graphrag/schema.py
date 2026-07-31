"""
schema de extração estruturada para o grafo de conhecimento da LGPD.

Sgue a abordagem do Capítulo 6 de "Essential GraphRAG" (Bratanič & Hane):
construção direta de KG via LLM com structured Outputs, com um schema
fechado (pydantic) em vez do pipeline de extração + sumarização + detecção
de comunidades do Capítulo 7 (MS GraphRAG),desnecessário para um corpus
pequeno e fechado como a LGPD (cerca de 80 artigos).

"""

from pydantic import BaseModel, Field


class EntidadeMencionada(BaseModel):
    """uma entidade jurídica mencionada ou definida no artigo"""

    nome: str = Field(
        ...,
        description=(
            "Nome curto e canônico da entidade, no formato como normalmente "
            "aparece na LGPD (ex.: 'consentimento', 'dado pessoal sensível', "
            "'encarregado'). Evite variações redundantes do mesmo conceito."
        ),
    )
    descricao: str = Field(
        ...,
        description=(
            "Descrição objetiva de 1-2 frases de como o artigo trata essa "
            "entidade, baseada exclusivamente no texto do artigo."
        ),
    )


class ArticleExtraction(BaseModel):
    """
    Extraçao struturada de um único artigo da LGPD

    cada lista é opcional (pode vir vazia), a maioria dos artigos não toca
    em todos os tipos de entidade
    """

    conceitos: list[EntidadeMencionada] = Field(
        default_factory=list,
        description="Definições ou conceitos jurídicos-chave introduzidos ou explicados neste artigo (ex.: dado pessoal, tratamento, anonimização).",
    )
    principios: list[EntidadeMencionada] = Field(
        default_factory=list,
        description="Princípios que regem o tratamento de dados estabelecidos neste artigo (ex.: finalidade, necessidade, transparência).",
    )
    direitos: list[EntidadeMencionada] = Field(
        default_factory=list,
        description="Direitos do titular dos dados estabelecidos ou detalhados neste artigo (ex.: acesso, correção, portabilidade).",
    )
    obrigacoes: list[EntidadeMencionada] = Field(
        default_factory=list,
        description="Obrigações impostas a controladores/operadores neste artigo (ex.: relatório de impacto, notificação de incidente).",
    )
    bases_legais: list[EntidadeMencionada] = Field(
        default_factory=list,
        description="Bases legais para tratamento de dados tratadas neste artigo (ex.: consentimento, legítimo interesse, cumprimento de obrigação legal).",
    )
    agentes: list[EntidadeMencionada] = Field(
        default_factory=list,
        description="Agentes/papéis mencionados neste artigo (ex.: controlador, operador, titular, ANPD, encarregado).",
    )
    sancoes: list[EntidadeMencionada] = Field(
        default_factory=list,
        description="Sanções ou penalidades administrativas tratadas neste artigo (ex.: multa, advertência, bloqueio).",
    )


# mapeia cada campo do schema para o verbo de relação usado no grafo
# (Article -[verbo]-> Entidade), e para o "tipo" armazenado no nó da entidade
FIELD_TO_RELATION = {
    "conceitos": ("DEFINE", "Conceito"),
    "principios": ("ESTABELECE", "Principio"),
    "direitos": ("ESTABELECE", "Direito"),
    "obrigacoes": ("ESTABELECE", "Obrigacao"),
    "bases_legais": ("ESTABELECE", "BaseLegal"),
    "agentes": ("MENCIONA", "Agente"),
    "sancoes": ("ESTABELECE", "Sancao"),
}

SYSTEM_MESSAGE = """Você é um especialista em direito de proteção de dados extraindo \
informação estruturada de artigos da Lei Geral de Proteção de Dados (LGPD - Lei nº \
13.709/2018) para construir um grafo de conhecimento.

Para cada artigo fornecido, identifique as entidades jurídicas relevantes (conceitos, \
princípios, direitos, obrigações, bases legais, agentes e sanções) que aparecem \
EXPLICITAMENTE no texto. Não infira informação que não esteja no artigo. Se um artigo \
não tratar de um determinado tipo de entidade, retorne a lista vazia para esse campo.

Use nomes curtos e canônicos para as entidades, de forma que o mesmo conceito \
mencionado em artigos diferentes (ex.: "consentimento" e "consentimento do titular") \
receba sempre o mesmo nome, isso é essencial para conectar o grafo corretamente."""
