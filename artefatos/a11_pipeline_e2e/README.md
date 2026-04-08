# A11 — Pipeline E2E

Entrega oficial do pipeline end-to-end do projeto SpectraAI.
O fluxo consolidado parte do dataset pronto, executa treino, avaliacao,
inferencia e exporta artefatos padronizados dentro desta pasta.

## Estrutura

```text
a11_pipeline_e2e/
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

Os caminhos sao definidos em `config.yaml` e resolvidos a partir do proprio
arquivo de configuracao.

## Execucao oficial

Instale as dependencias:

```bash
pip install -r artefatos/a11_pipeline_e2e/requirements.txt
```

Execute o pipeline:

```bash
python artefatos/a11_pipeline_e2e/main.py --config artefatos/a11_pipeline_e2e/config.yaml
```

Modo rapido:

```bash
python artefatos/a11_pipeline_e2e/main.py \
  --config artefatos/a11_pipeline_e2e/config.yaml \
  --limit-samples 64
```

## Flags uteis

- `--limit-samples`: reduz o dataset para smoke tests e validacao rapida
- `--skip-train`: reutiliza `outputs/models/best_model.keras` ja salvo
- `--skip-inference`: pula exportacao de predicoes e visualizacoes finais
- `--output-dir`: redireciona os outputs para outro diretorio

## Saidas geradas

Uma execucao completa gera no minimo:

- `outputs/metrics/summary.json`
- `outputs/metrics/summary.csv`
- `outputs/models/best_model.keras`
- `outputs/models/history.json`
- `outputs/predictions/test_predictions.csv`
- `outputs/visualizations/confusion_matrix.png`
- `outputs/visualizations/roc_pr_curves.png`

## Modelo final

O A11 usa Transfer Learning com MobileNetV2 como modelo oficial do pipeline,
reaproveitando os componentes ja implementados no codigo principal do projeto.

## Notebook

O notebook `notebooks/a11_pipeline_e2e.ipynb` foi mantido como apoio
narrativo. A reproducao oficial da entrega, no entanto, deve ser feita pelo
`main.py`.
