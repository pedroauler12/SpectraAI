# A09 - Interpretabilidade e Visualização

Arquivos deste artefato:

- `artefatos/a09_interpretabilidade_visualizacao/a09_interpretabilidade_visualizacao.ipynb`
- `artefatos/a09_interpretabilidade_visualizacao/README.md`

## Objetivo

Este artefato analisa o comportamento do modelo de **transfer learning** treinado
no A08 e atende explicitamente aos **itens 1, 2, 3, 4 e 5** da rubrica do A09.
O foco permanece na explicação dos resultados, na interpretabilidade e no apoio
à tomada de decisão, sem realizar novo treinamento do modelo.

## Mapeamento da rubrica

**Item 1 - Geração de predições e outputs do modelo**

O notebook reconstrói o split do A08, executa inferências em validação e teste,
ajusta `threshold_f1` na validação e salva outputs tabulares para análise:
`val_predictions.csv`, `test_predictions.csv` e
`threshold_selection_summary.csv`.

**Item 2 - Visualizações quantitativas de desempenho**

São geradas curvas de treino, matrizes de confusão, ROC/PR, sweep de thresholds,
distribuições de probabilidade, boxplots e tabelas de métricas agregadas e por
classe.

**Item 3 - Visualizações espaciais ou mapas de predição**

As predições são conectadas a coordenadas reais das amostras ASTER para gerar
mapas geoespaciais de probabilidade, acertos/erros em `threshold=0.5` e em
`threshold_f1`, além de visualizações complementares de densidade e confiança.

**Item 4 - Técnicas de interpretabilidade**

O notebook aplica **Grad-CAM** para explicar decisões do modelo em exemplos
corretos e incorretos selecionados por critério explícito, incluindo comparação
entre classes e uma leitura das limitações do método em dados multibanda.

**Item 5 - Interpretação crítica das visualizações**

Cada bloco principal do notebook traz interpretação textual obrigatória. A
conclusão final consolida achados empíricos, limitações, implicações práticas e
próximos passos recomendados.

## Dependência do A08

O notebook não retreina o modelo. Ele reutiliza diretamente os artefatos
gerados em `outputs/a08_transfer_learning/`:

- `history.csv`
- `best_model.keras`

Também reconstrói o mesmo split e o mesmo `tf.data.Dataset` com:

- `src/models/cnn_data_prep.py`
- `src/models/cnn_tf_data_pipeline.py`

## Fluxo executado no notebook

Ao executar o notebook, são realizados os seguintes passos:

1. leitura do histórico consolidado do A08;
2. reconstrução do split treino/validação/teste e carregamento do modelo salvo;
3. inferência em validação para ajuste de `threshold_f1`;
4. inferência em teste para comparação entre `threshold=0.5` e `threshold_f1`;
5. geração de visualizações quantitativas, espaciais e de interpretabilidade;
6. registro de interpretação textual após cada bloco principal;
7. salvamento dos outputs tabulares e gráficos em
   `outputs/a09_interpretabilidade_visualizacao/`.

## Outputs gerados

Os principais arquivos salvos ficam em:

- `outputs/a09_interpretabilidade_visualizacao/val_predictions.csv`
- `outputs/a09_interpretabilidade_visualizacao/test_predictions.csv`
- `outputs/a09_interpretabilidade_visualizacao/threshold_selection_summary.csv`
- `outputs/a09_interpretabilidade_visualizacao/test_metrics_comparison.csv`
- `outputs/a09_interpretabilidade_visualizacao/per_class_metrics.csv`
- `outputs/a09_interpretabilidade_visualizacao/geospatial_predictions.csv`
- `outputs/a09_interpretabilidade_visualizacao/training_curves_from_history.png`
- `outputs/a09_interpretabilidade_visualizacao/confusion_matrix_threshold_05.png`
- `outputs/a09_interpretabilidade_visualizacao/confusion_matrix_threshold_f1.png`
- `outputs/a09_interpretabilidade_visualizacao/confusion_matrix_norm_threshold_05.png`
- `outputs/a09_interpretabilidade_visualizacao/confusion_matrix_norm_threshold_f1.png`
- `outputs/a09_interpretabilidade_visualizacao/probability_distributions.png`
- `outputs/a09_interpretabilidade_visualizacao/roc_pr_curves.png`
- `outputs/a09_interpretabilidade_visualizacao/threshold_sweep_metrics.png`
- `outputs/a09_interpretabilidade_visualizacao/probability_boxplot.png`
- `outputs/a09_interpretabilidade_visualizacao/spatial_probability_map.png`
- `outputs/a09_interpretabilidade_visualizacao/spatial_outcome_map.png`
- `outputs/a09_interpretabilidade_visualizacao/spatial_outcome_map_threshold_f1.png`
- `outputs/a09_interpretabilidade_visualizacao/spatial_probability_hexbin.png`
- `outputs/a09_interpretabilidade_visualizacao/spatial_confidence_bubble_map.png`
- `outputs/a09_interpretabilidade_visualizacao/spatial_probability_by_outcome.png`
- `outputs/a09_interpretabilidade_visualizacao/sample_chips_marked_grid.png`
- `outputs/a09_interpretabilidade_visualizacao/gradcam_por_outcome.png`
- `outputs/a09_interpretabilidade_visualizacao/gradcam_comparativo.png`
- `outputs/a09_interpretabilidade_visualizacao/interpretabilidade_summary.json`

## Demo complementar em Streamlit

Além do notebook oficial, a entrega inclui uma demo local em Streamlit:

- `apps/a09_geo_demo.py`

Essa aplicação permite:

- clicar em qualquer ponto do mapa;
- buscar um granule ASTER via NASA EarthData;
- recortar e empilhar um chip multibanda;
- rodar inferência com o modelo treinado no A08;
- comparar threshold otimizado (`threshold_f1`) com o corte padrão `0.5`;
- visualizar a probabilidade prevista;
- comparar preview RGB natural e composição false-color do chip;
- sinalizar quando a qualidade da cena estiver degradada por nuvem, sombra ou
  baixa variação.

Execução local:

```bash
streamlit run apps/a09_geo_demo.py
```

> A demo depende de credenciais EarthData válidas e acesso à rede. Ela funciona
> como extensão prática do artefato, mas o notebook continua sendo a entrega
> formal e reprodutível do A09.

## Código reutilizável adicionado

Para sustentar o item 3 e a demo, o projeto inclui:

- `src/inference/transfer_geo_inference.py`
  - reconstrução do normalizador do A08;
  - inferência sobre chips ASTER multibanda;
  - fluxo de busca e recorte via EarthData;
  - geração de preview RGB e false-color para visualização;
  - avaliação simples de qualidade do chip para apoio a interpretação.

## Compatibilidade de ambiente

Assim como no A08, este notebook foi pensado para rodar em ambiente com
TensorFlow/Keras disponível, preferencialmente:

- Python `3.10`, `3.11` ou `3.12`
- ou Google Colab

Se o ambiente local não tiver wheels compatíveis de TensorFlow, execute o
notebook no Colab mantendo os mesmos caminhos de dados e outputs.

Para a demo Streamlit, o ambiente também precisa de:

- `streamlit`
- `folium`
- `streamlit-folium`
- `openpyxl`
