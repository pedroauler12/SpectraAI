# A01 — Entendimento do Problema & Dados (EDA + Pipeline)

## Objetivo

Estabelecer o entendimento do problema de prospecção de Terras Raras da Frontera Minerals, inventariar os dados disponíveis (imagens ASTER), formular hipóteses geológicas para a modelagem e construir o pipeline inicial de pré-processamento dos dados espectrais.

## O que foi feito

- **Entendimento do problema:** contexto do parceiro (Frontera Minerals), definição do problema, objetivo do projeto e MVP
- **Mapeamento de stakeholders** e levantamento de necessidades
- **Escolha de métricas** e justificativa para avaliação do modelo
- **Análise do sensor ASTER:** descrição técnica, comparativo espectral, relevância das bandas SWIR/VNIR para Terras Raras
- **Hipóteses geológicas:** três hipóteses sobre assinaturas espectrais de minerais de argila, carbonatos e organização espacial de alteração
- **Inventário dos dados:** origem, critérios de seleção temporal e espectral, estrutura de diretórios, descrição das 14 bandas ASTER, amostras disponíveis e limitações do dataset
- **Pipeline inicial de pré-processamento:** leitura de tiles ASTER, extração de bandas e geração de chips georreferenciados

## Principais conclusões

- O sensor ASTER (14 bandas VNIR/SWIR/TIR) é o mais adequado para detecção de minerais associados a Terras Raras no contexto do projeto
- As bandas SWIR (B04–B09) são as mais relevantes para discriminar minerais de argila e carbonatos indicativos de prospectividade
- O dataset apresenta limitações importantes: cobertura vegetal intensa, resolução espacial heterogênea entre subsistemas e anomalias conhecidas no subsistema SWIR
- A abordagem pixel-level com chips extraídos de tiles ASTER é viável e reprodutível

## Como executar

```bash
# Com ambiente ativo e dependências instaladas:
jupyter notebook artefatos/a01_eda_pipeline/a01_eda_pipeline.ipynb
```

> **Pré-requisitos:** dados ASTER disponíveis no caminho configurado em `src/tiles/config.py`. Ver seção "Como Executar" no README raiz para configuração do ambiente.
