# CNN Multissatelite

## Objetivo
A pasta contem dois notebooks principais:

1. `cnn_multissatelite.ipynb`
   - Primeiro modelo CNN por satelite (Sentinel-2, Landsat 8/9, MODIS), executando 5 etapas completas por vez.

2. `cnn_experimentos_multissatelite.ipynb`
   - Avaliacao experimental sistematica com protocolo controlado, multiplos experimentos,
     comparacao consolidada, ablacoes e interpretacao critica.

Ambos executam satelite por satelite para evitar estouro de RAM.

## Arquivos
- `notebooks/cnn_multissatelite/cnn_multissatelite.ipynb`
- `notebooks/cnn_multissatelite/cnn_experimentos_multissatelite.ipynb`
- `notebooks/cnn_multissatelite/README.md`

## Requisitos
- Python 3.10+
- TensorFlow instalado
- Dependencias do projeto (`requirements-dev.txt`)
- Parquets em formato longo por sensor

## Entradas esperadas
- `data/parquet/sentinel_completo.parquet`
- `data/parquet/landsat89_completo.parquet`
- `data/parquet/modis_completo.parquet`
- `data/extracted_codes.json`

## Como executar
Na raiz do repositorio:

```bash
jupyter notebook notebooks/cnn_multissatelite/cnn_multissatelite.ipynb
```

ou

```bash
jupyter notebook notebooks/cnn_multissatelite/cnn_experimentos_multissatelite.ipynb
```

Execute as celulas em ordem.

## Saidas principais
### Notebook 1 (modelo CNN inicial)
- `outputs/a05_cnn_multissatelite/<sensor>/...`
- `outputs/a05_cnn_multissatelite/cnn_multissatelite_comparison.csv`
- `outputs/a05_cnn_multissatelite/cnn_multissatelite_analysis.txt`

### Notebook 2 (avaliacao experimental)
- `outputs/a06_cnn_experimentos/experimental_protocol.json`
- `outputs/a06_cnn_experimentos/all_experiment_results.csv`
- `outputs/a06_cnn_experimentos/ablation_results.csv` (quando houver)
- `outputs/a06_cnn_experimentos/critical_analysis.txt`
- `outputs/a06_cnn_experimentos/run_summary.json`

## Reprodutibilidade
- Seed fixa (`SEED = 42`)
- Split estratificado treino/validacao/teste
- Comparacao justa com split fixo por satelite no notebook de experimentos
- Normalizacao por canal ajustada apenas no treino
- Checkpoint do melhor modelo por `val_loss`

## Controle de memoria no notebook de experimentos
- Por padrao, `cnn_experimentos_multissatelite.ipynb` roda em `RUN_MODE = "single"`.
- Nesse modo, executa **1 satelite + 1 experimento por vez**.
- Ajuste `SINGLE_SENSOR_KEY` e `SINGLE_EXPERIMENT_NAME` para cada rodada.
- Os resultados sao acumulados em `outputs/a06_cnn_experimentos/all_experiment_results.csv`.
