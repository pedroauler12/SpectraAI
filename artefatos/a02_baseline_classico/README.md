# Artefato A02 - Baseline Classico

## Objetivo
Implementar e avaliar modelos classicos de Machine Learning como baseline para classificacao, usando o dataset preparado no Artefato A01.

Este artefato estabelece referencias quantitativas para comparacao com modelos de Deep Learning nas proximas sprints.

## Arquivos do artefato
- `artefatos/a02_baseline_classico/a02_baseline_classico.ipynb`
- `artefatos/a02_baseline_classico/README.md`

## Requisitos
- Python 3.10+
- Jupyter Notebook
- Dependencias instaladas (recomendado via `requirements-dev.txt` na raiz do projeto)

Instalacao recomendada:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
```

## Entradas necessarias
O notebook espera os seguintes arquivos na **raiz do repositorio**:

- `data/extracted_codes.json`
- `data/pixels_dataset.csv`

Se `pixels_dataset.csv` ainda nao existir, gere-o na etapa de preparacao de dados (A01 / pipeline de extracao de pixels).

## Como executar
Na raiz do projeto:

```bash
jupyter notebook artefatos/a02_baseline_classico/a02_baseline_classico.ipynb
```

Execute as celulas em ordem, sem pular etapas.

## O que o notebook executa
1. Carregamento e preparacao dos dados para modelagem.
2. Split treino/validacao/teste com controle por `image_id`.
3. Treino de tres modelos classicos:
   - Random Forest
   - SVM
   - Regressao Logistica
4. Selecao de hiperparametros (CV) e calibracao de threshold na validacao.
5. Avaliacao final em teste com metricas padronizadas.
6. Analise de erros (FP/FN) e discussao critica.
7. Export dos resultados para `outputs/`.

## Saidas geradas
Ao final da execucao, sao salvos:

- `outputs/a02_baseline_classico/summary_metrics.csv`
- `outputs/a02_baseline_classico/full_results.json`
- `outputs/a02_baseline_classico/best_model_errors_test.csv`

## Reprodutibilidade
- Seed fixa (`SEED = 42`) no notebook.
- Protocolo unico para todos os modelos (mesmo split, mesma estrategia de selecao e avaliacao).
- Teste final isolado para estimativa de generalizacao.

