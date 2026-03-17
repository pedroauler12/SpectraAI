# A08 - Transfer Learning

Arquivos deste artefato:

- `artefatos/a08_transfer_learning/a08_transfer_learning.ipynb`
- `artefatos/a08_transfer_learning/README.md`

## Objetivo

Evoluir o pipeline convolucional com **transfer learning** e **data augmentation**,
mantendo consistencia metodologica com os artefatos anteriores.

Este notebook foi preparado para usar a mesma base de dados dos notebooks:

- `artefatos/a02_baseline_classico/a02_baseline_classico.ipynb`
- `artefatos/a05_cnn_simples/a05_cnn_simples.ipynb`

## O que foi consolidado no codigo reutilizavel

As etapas de dados deixaram de ficar espalhadas no notebook e passaram a usar funcoes
do proprio projeto:

- `src/models/cnn_data_prep.py`
  - split estratificado por `image_id`;
  - conversao de `pixel_*` para tensor 4D;
  - separacao `treino/validacao/teste` sem vazamento entre imagens.
- `src/models/cnn_tf_data_pipeline.py`
  - `resize` espacial para o backbone;
  - normalizacao ajustada apenas no treino e reaplicada em validacao/teste;
  - `tf.data` com augmentacao apenas no treino;
  - helper para montar `train_ds`, `val_ds` e `test_ds`.

## Decisao de entrada para transfer learning

Como os chips ASTER possuem `9` bandas e o backbone pre-treinado escolhido no notebook
(`MobileNetV2`) espera `3` canais, o pipeline usa:

- entrada `9` canais preservada ate o modelo;
- camada adaptadora `1x1` para projetar `9 -> 3`;
- backbone pre-treinado congelado parcialmente, com fine-tuning nas ultimas camadas.

## Saidas esperadas

Ao executar o notebook, os principais arquivos ficam em:

- `outputs/a08_transfer_learning/history.csv`
- `outputs/a08_transfer_learning/transfer_learning_metrics.json`
- `outputs/a08_transfer_learning/comparison_with_previous_artifacts.csv`

## Regenerar o notebook

Se precisar recriar o `.ipynb`, execute:

```bash
python scripts/generate_a08_transfer_learning_notebook.py
```

## Compatibilidade de ambiente

O notebook usa TensorFlow/Keras e foi pensado para ambientes com:

- Python `3.10`, `3.11` ou `3.12`
- ou Google Colab

No ambiente local atual com Python `3.14`, `pip install tensorflow` nao encontra wheel
compativel. Nessa situacao, rode o notebook no Colab ou crie um ambiente separado com
Python `3.11`/`3.12`.
