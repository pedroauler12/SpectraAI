# CNN Multissatelite

## Objetivo
Implementar o primeiro modelo CNN do projeto em um notebook unico, com execucao por satelite:
1. Sentinel-2 (itens 1 a 5)
2. Landsat 8/9 (itens 1 a 5)
3. MODIS (itens 1 a 5)

Esse fluxo evita carregar todos os datasets ao mesmo tempo e reduz risco de estouro de RAM.

## Arquivos
- `notebooks/cnn_multissatelite/cnn_multissatelite.ipynb`
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

Execute as celulas em ordem. Cada bloco de satelite faz as 5 etapas completas e libera memoria ao final.

## Saidas geradas
- `outputs/a05_cnn_multissatelite/<sensor>/...`
- `outputs/a05_cnn_multissatelite/cnn_multissatelite_comparison.csv`
- `outputs/a05_cnn_multissatelite/cnn_multissatelite_analysis.txt`
- `outputs/a05_cnn_multissatelite/run_summary.json`

## Reprodutibilidade
- Seed fixa (`SEED = 42`)
- Split estratificado treino/validacao/teste
- Normalizacao por canal ajustada apenas no treino
- Checkpoint do melhor modelo por `val_loss`
