# A11 — Pipeline E2E

Entrega oficial do pipeline end-to-end do projeto SpectraAI. Este artefato
consolida preparacao de dados, treinamento, avaliacao em teste, inferencia e
geracao automatica de resultados em um fluxo unico e reproduzivel.

## Objetivo do artefato

O A11 atende ao barema da entrega com os seguintes compromissos:

- integrar o fluxo completo em um unico entrypoint;
- permitir reproducao por um comando oficial;
- manter o codigo modular e organizado dentro da pasta do artefato;
- gerar saidas padronizadas automaticamente;
- documentar instalacao, execucao e reproducao dos experimentos.

## Estrutura

```text
a11_pipeline_e2e/
├── __main__.py
├── config.yaml
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── outputs/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   └── visualizations/
└── notebooks/
    └── a11_pipeline_e2e.ipynb
```

## Entradas exigidas

O pipeline parte do dataset pronto ja existente no projeto:

- `data/pixels_dataset.csv`
- `data/extracted_codes.json`

Os caminhos sao definidos em `config.yaml` e resolvidos a partir da propria
pasta do artefato.

## Ambiente testado

- Python `3.11`
- execucao oficial via modulo: `python3 -m artefatos.a11_pipeline_e2e`

## Instalacao

Crie um ambiente virtual e instale as dependencias do artefato:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r artefatos/a11_pipeline_e2e/requirements.txt
```

## Comando oficial de execucao

Execute o pipeline completo com:

```bash
python3 -m artefatos.a11_pipeline_e2e \
  --config artefatos/a11_pipeline_e2e/config.yaml
```

## Modo rapido

Para smoke tests e validacao rapida:

```bash
python3 -m artefatos.a11_pipeline_e2e \
  --config artefatos/a11_pipeline_e2e/config.yaml \
  --limit-samples 64
```

## Flags publicas

- `--config`: caminho do arquivo `config.yaml`
- `--limit-samples`: reduz o dataset para smoke tests
- `--skip-train`: reutiliza `outputs/models/best_model.keras` ja salvo
- `--skip-inference`: pula exportacao de predicoes e visualizacoes finais
- `--output-dir`: salva os outputs em outro diretorio

## Resultados gerados automaticamente

Uma execucao completa gera, de forma padronizada:

- `outputs/metrics/summary.json`
- `outputs/metrics/summary.csv`
- `outputs/models/best_model.keras`
- `outputs/models/history.json`
- `outputs/predictions/test_predictions.csv`
- `outputs/visualizations/confusion_matrix.png`
- `outputs/visualizations/roc_pr_curves.png`

O `summary.json` e o `summary.csv` exportam o resumo tecnico da execucao,
incluindo:

- `evaluation_split`
- `test_accuracy`
- `test_precision`
- `test_recall`
- `test_f1`
- `test_balanced_accuracy`
- `test_roc_auc`
- `test_pr_auc`

## Reproducao dos experimentos

Para reproduzir os experimentos do A11:

1. instale as dependencias com o mesmo interpretador usado na execucao;
2. confira se `data/pixels_dataset.csv` e `data/extracted_codes.json` estao presentes;
3. rode o comando oficial do artefato;
4. verifique os arquivos gerados em `artefatos/a11_pipeline_e2e/outputs/`.

## Politica de outputs

Os arquivos em `outputs/` sao gerados automaticamente e podem ser
reconstruidos. Por isso, o repositorio mantem apenas os `.gitkeep` e nao usa
os outputs como fonte primaria de evidencia da entrega.

## Modelo final

O pipeline oficial usa Transfer Learning com MobileNetV2, reaproveitando os
componentes principais do projeto e padronizando a avaliacao final no conjunto
de teste.

## Notebook

O notebook `notebooks/a11_pipeline_e2e.ipynb` foi mantido como apoio
narrativo. A reproducao oficial da entrega deve ser feita pela CLI, usando o
comando por modulo documentado acima.

## Troubleshooting

- `ModuleNotFoundError: artefatos`: execute via `python3 -m artefatos.a11_pipeline_e2e`.
- `FileNotFoundError` para dataset ou labels: confirme os caminhos definidos em `config.yaml`.
- Falta de dependencias: reinstale com `python3 -m pip install -r artefatos/a11_pipeline_e2e/requirements.txt`.
