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

## Parametros do config.yaml

| Parametro | Valor padrao | Descricao |
|---|---|---|
| `seed` | `42` | Semente global para reprodutibilidade (numpy, tensorflow, python) |
| `data.image_size` | `[128, 128]` | Resolucao espacial dos patches ASTER em pixels |
| `data.num_bands` | `9` | Numero de bandas multiespectrais do sensor ASTER |
| `data.normalization_method` | `zscore` | Normalizacao por z-score (media=0, desvio=1) por banda |
| `data.test_size` | `0.2` | Fracao do dataset reservada para teste final (20%) |
| `data.val_size` | `0.2` | Fracao do treino reservada para validacao durante o treinamento |
| `model.backbone` | `mobilenetv2` | Rede pre-treinada usada como extrator de features |
| `model.resize_to` | `[160, 160]` | Resolucao de entrada esperada pelo MobileNetV2 |
| `model.dropout_rate` | `0.25` | Taxa de dropout aplicada antes da camada de classificacao |
| `model.fine_tune_last_layers` | `20` | Quantas camadas finais do backbone sao descongeladas na fase 2 |
| `training.head_epochs` | `6` | Epocas maximas da fase 1 (somente cabeca treinada) |
| `training.fine_tune_epochs` | `12` | Epocas maximas da fase 2 (backbone parcialmente descongelado) |
| `training.head_learning_rate` | `1e-4` | Taxa de aprendizado da fase 1 |
| `training.fine_tune_learning_rate` | `1e-5` | Taxa de aprendizado da fase 2 (menor para nao destruir features pre-treinadas) |
| `evaluation.threshold_default` | `0.5` | Limiar de decisao binaria: probabilidade >= threshold → classe positiva |

## Entradas exigidas

O pipeline parte do dataset pronto ja existente no projeto:

- `data/pixels_dataset.csv`
- `data/extracted_codes.json`

Os caminhos sao definidos em `config.yaml` e resolvidos a partir da propria
pasta do artefato.

## Como obter os dados

Os arquivos de entrada **nao estao versionados no repositorio** por conta do
tamanho (137 MB). Para reproduzir o experimento:

1. Obtenha o `pixels_dataset.csv` com o membro do grupo responsavel pelo
   pipeline de dados (sprint A05/A06), ou reconstrua-o a partir dos
   notebooks de pre-processamento em `notebooks/`.
2. Coloque os arquivos em `data/` na raiz do repositorio antes de executar.

Os caminhos esperados sao:

```
data/pixels_dataset.csv         # dataset multispectral ASTER (295 amostras, 9 bandas, 128x128 px)
data/extracted_codes.json       # rotulos binarios (positivo/negativo para terras raras)
```

## Diretorio de execucao

O comando oficial deve ser executado **a partir da raiz do repositorio**
(onde a pasta `artefatos/` esta visivel). Os caminhos relativos definidos
em `config.yaml` pressupõem esse diretorio de trabalho:

```bash
cd /caminho/para/g01        # raiz do repositorio
python3 -m artefatos.a11_pipeline_e2e --config artefatos/a11_pipeline_e2e/config.yaml
```

Executar de outro diretorio resultara em `FileNotFoundError` ao resolver os
caminhos do dataset.

## Requisitos de hardware

| Recurso | Minimo recomendado |
|---|---|
| RAM | 8 GB (dataset 137 MB + overhead TensorFlow) |
| CPU | Qualquer processador moderno |
| GPU | Opcional (CUDA acelera o treinamento) |
| Armazenamento | ~500 MB livres para outputs |
| Python | 3.11 (versao testada) |

Tempo estimado de execucao completa:
- **CPU:** 3–10 minutos
- **GPU (CUDA):** 1–2 minutos

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
- `outputs/visualizations/training_curves.png`

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

## Saidas esperadas

Apos uma execucao completa com o dataset completo (295 amostras), os valores
de referencia obtidos na ultima execucao validada sao:

```json
{
  "test_accuracy": 0.881,
  "test_precision": 0.833,
  "test_recall": 0.870,
  "test_f1": 0.851,
  "test_balanced_accuracy": 0.879,
  "test_roc_auc": 0.935,
  "test_pr_auc": 0.875,
  "n_test": 59,
  "total_epochs": 13
}
```

Variacoes menores nos valores sao esperadas entre execucoes por diferencas
de hardware (ordenacao de operacoes de ponto flutuante), mas devem ser
inferiores a 2% nas metricas principais com `seed: 42`.

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
