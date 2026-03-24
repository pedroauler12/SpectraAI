# A09 - Interpretabilidade e Visualizacao

Arquivos deste artefato:

- `artefatos/a09_interpretabilidade_visualizacao/a09_interpretabilidade_visualizacao.ipynb`
- `artefatos/a09_interpretabilidade_visualizacao/README.md`

## Objetivo

Este artefato analisa o comportamento do modelo de **transfer learning** treinado
no A08 por meio de visualizacoes quantitativas e interpretacao textual dos
resultados. O foco principal desta entrega cobre os **itens 2 e 3** da rubric:

- curvas de treino (`loss` e `accuracy`);
- matrizes de confusao em dois thresholds;
- distribuicoes de probabilidade por classe;
- leitura critica dos padroes observados.
- visualizacoes espaciais/geoespaciais com coordenadas reais das amostras ASTER.

## Dependencia do A08

O notebook nao retreina o modelo. Ele reutiliza diretamente os artefatos
gerados em `outputs/a08_transfer_learning/`:

- `history.csv`
- `best_model.keras`

Tambem reconstrui o mesmo split e o mesmo `tf.data.Dataset` com:

- `src/models/cnn_data_prep.py`
- `src/models/cnn_tf_data_pipeline.py`

## O que o notebook entrega nos itens 2 e 3

Ao executar o notebook, sao produzidos:

- leitura do historico consolidado do A08;
- inferencia em validacao para ajuste de `threshold_f1`;
- inferencia em teste com comparacao entre `threshold=0.5` e `threshold_f1`;
- matrizes de confusao absolutas e normalizadas;
- grafico de distribuicao de `P(classe positiva)` por classe real;
- tabela comparativa de metricas no conjunto de teste;
- merge georreferenciado entre predições e pontos reais do banco;
- mapa espacial das probabilidades previstas;
- mapa espacial de acertos e erros (`TP`, `TN`, `FP`, `FN`);
- interpretacao textual apos cada bloco principal.

## Demo complementar em Streamlit

Além do notebook oficial, a entrega inclui uma demo local em Streamlit:

- `apps/a09_geo_demo.py`

Essa aplicacao permite:

- clicar em qualquer ponto do mapa;
- buscar um granule ASTER via NASA EarthData;
- recortar e empilhar um chip multibanda;
- rodar inferencia com o modelo treinado no A08;
- visualizar a probabilidade prevista e uma composicao false-color do chip.

Execucao local:

```bash
streamlit run apps/a09_geo_demo.py
```

> A demo depende de credenciais EarthData validas e acesso a rede. Ela funciona
> como extensao prática do artefato, mas o notebook continua sendo a entrega
> formal e reprodutível do A09.

## Outputs gerados

Os principais arquivos salvos ficam em:

- `outputs/a09_interpretabilidade_visualizacao/training_curves_from_history.png`
- `outputs/a09_interpretabilidade_visualizacao/confusion_matrix_threshold_05.png`
- `outputs/a09_interpretabilidade_visualizacao/confusion_matrix_threshold_f1.png`
- `outputs/a09_interpretabilidade_visualizacao/confusion_matrix_norm_threshold_05.png`
- `outputs/a09_interpretabilidade_visualizacao/confusion_matrix_norm_threshold_f1.png`
- `outputs/a09_interpretabilidade_visualizacao/probability_distributions.png`
- `outputs/a09_interpretabilidade_visualizacao/test_predictions.csv`
- `outputs/a09_interpretabilidade_visualizacao/test_metrics_comparison.csv`
- `outputs/a09_interpretabilidade_visualizacao/geospatial_predictions.csv`
- `outputs/a09_interpretabilidade_visualizacao/spatial_probability_map.png`
- `outputs/a09_interpretabilidade_visualizacao/spatial_outcome_map.png`

## Codigo reutilizavel adicionado

Para sustentar a entrega do item 3 e a demo, o projeto passa a incluir:

- `src/inference/transfer_geo_inference.py`
  - reconstrucao do normalizador do A08;
  - inferencia sobre chips ASTER multibanda;
  - fluxo de busca e recorte via EarthData;
  - geracao de preview false-color para visualizacao.

## Compatibilidade de ambiente

Assim como no A08, este notebook foi pensado para rodar em ambiente com
TensorFlow/Keras disponivel, preferencialmente:

- Python `3.10`, `3.11` ou `3.12`
- ou Google Colab

Se o ambiente local nao tiver wheels compativeis de TensorFlow, execute o
notebook no Colab mantendo os mesmos caminhos de dados e outputs.

Para a demo Streamlit, o ambiente tambem precisa de:

- `streamlit`
- `folium`
- `streamlit-folium`
- `openpyxl`
