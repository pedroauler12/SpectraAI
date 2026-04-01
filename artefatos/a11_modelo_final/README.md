# A11 — Modelo Final

Artefato de entrega final do projeto de classificação de imagens ASTER.
Consolida o pipeline completo — do pré-processamento à inferência — de forma modular e reprodutível.

## Estrutura de pastas

```
a11_modelo_final/
├── config.yaml              # Parâmetros globais: caminhos, seed, hiperparâmetros
├── requirements.txt         # Dependências Python
├── .gitignore
│
├── src/                     # Código-fonte modular (pacote Python)
│   ├── __init__.py
│   ├── preprocessing/       # Leitura, normalização e preparo dos tiles ASTER
│   │   └── __init__.py
│   ├── training/            # Loop de treinamento, callbacks e checkpoints
│   │   └── __init__.py
│   ├── evaluation/          # Métricas, curvas ROC/PR, matrizes de confusão
│   │   └── __init__.py
│   └── inference/           # Predição em novos dados e exportação de resultados
│       └── __init__.py
│
├── data/
│   ├── raw/                 # Dados originais (tiles .tif / .npy) — não versionados
│   └── processed/           # Arrays normalizados prontos para treino — não versionados
│
├── outputs/
│   ├── metrics/             # CSVs e JSONs com métricas por época e experimento
│   ├── models/              # Pesos salvos (.h5 / SavedModel) — não versionados
│   ├── visualizations/      # Gráficos gerados (PNG/SVG)
│   └── predictions/         # CSVs de predições (val, test)
│
└── notebooks/               # Notebooks exploratórios e de análise
```

## Como usar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Ajuste os parâmetros em `config.yaml` conforme necessário (seed, caminhos, hiperparâmetros).

3. Execute os scripts na ordem:
   ```
   preprocessing → training → evaluation → inference
   ```

## Configuração central (`config.yaml`)

Todos os scripts importam `config.yaml` para garantir que seed, caminhos e
hiperparâmetros sejam consistentes entre etapas. Não codifique caminhos ou
parâmetros diretamente nos scripts — sempre leia do config.
