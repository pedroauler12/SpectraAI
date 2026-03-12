# Artefato A05 - CNN Simples

## Arquivos de entrega
- `artefatos/a05_cnn_simples/a05_cnn_simples.ipynb`
- `artefatos/a05_cnn_simples/README.md`

## Objetivo
Este artefato apresenta o primeiro modelo convolucional do projeto como **notebook principal de implementação**.

O arquivo `a05_cnn_simples.ipynb` reúne, no próprio artefato:
- preparação dos dados para CNN;
- implementação explícita da arquitetura convolucional;
- treinamento e validação do baseline original;
- avaliação quantitativa com comparação ao baseline MLP da Sprint 2;
- ampliação final por sensor;
- análise crítica consolidada.

## O que foi ajustado nesta versão
Esta versão foi reorganizada para evitar que o A05 pareça apenas um notebook de consolidação retrospectiva. O foco passou a ser:

- trazer o **código principal** de preparação, arquitetura e treino para dentro do próprio notebook;
- usar os arquivos de `outputs/` como evidência reprodutível do que foi executado;
- manter a ampliação final por sensor como extensão analítica do baseline original.

## Fontes incorporadas
As fontes abaixo foram usadas para reconstruir o artefato:

- `src/models/cnn_data_prep.py`
- `src/models/cnn_builder.py`
- `src/models/experiment_runner.py`
- `src/models/configs/baseline.yaml`
- `notebooks/cnn_preparacao_dados.ipynb`
- `notebooks/a05_cnn_teste.ipynb`
- `notebooks/a05b_cnn_experiments.ipynb`
- `notebooks/final/a05_cnn_simples_processo_completo.ipynb`
- `outputs/a04_cnn_data_prep/cnn_data_prep_summary.json`
- `outputs/a04_cnn_data_prep/cnn_normalizer_zscore.npz`
- `outputs/trained_models/experiments_log.csv`
- `outputs/trained_models/baseline_20260311_083736/history.json`
- `outputs/a06_avaliacao_experimental/e1_baseline_summary.json`
- `outputs/a03_mlp_baseline/mlp_baseline_metrics.json`
- `artefatos/a03_mlp_baseline/a03_mlp_baseline.ipynb`
- `outputs/final/`
- `outputs/final/a05_cnn_multissatelite/`

## Estrutura do notebook
O notebook final foi organizado em nove seções:

1. setup e carregamento dos artefatos;
2. escopo, fontes e critério de consolidação;
3. preparação dos dados para a CNN;
4. implementação da arquitetura CNN simples;
5. treinamento e validação do modelo;
6. avaliação quantitativa do baseline original;
7. ampliação final com CNN simples por sensor;
8. análise crítica consolidada;
9. checklist de entrega.

## Reexecução
O notebook abre com outputs já gravados no `.ipynb`.

Se for necessário regenerar:

```bash
python3 scripts/generate_a05_artifact_notebook.py artefatos/a05_cnn_simples/a05_cnn_simples.ipynb
python3 scripts/render_notebook_outputs.py artefatos/a05_cnn_simples/a05_cnn_simples.ipynb
```

## Dependências práticas
- Python 3.10+
- `pandas`
- `numpy`
- `matplotlib`
- `IPython`

`TensorFlow` não é obrigatório para revisar o notebook final nesta máquina. Quando ele não está disponível, o notebook continua exibindo:
- o código consolidado de preparação, arquitetura e treino;
- os históricos e resumos salvos em `outputs/`;
- os gráficos, tabelas e análises finais já incorporados ao `.ipynb`.

## Observações metodológicas
- O baseline CNN original aparece com métricas de validação interna.
- O baseline MLP salvo no projeto aparece com métricas de teste hold-out.
- A comparação inicial entre CNN e MLP deve ser lida como comparação indicativa de evolução.
- A leitura metodologicamente mais forte está na comparação final por sensor, baseada em `outputs/final/a05_cnn_multissatelite/`.

## Status da entrega
Com a versão atual, os cinco itens do enunciado ficam cobertos dentro da pasta `artefatos/a05_cnn_simples/`, com o notebook funcionando como artefato central da entrega.
