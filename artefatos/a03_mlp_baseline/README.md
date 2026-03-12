# A03 - Baseline Deep Learning (MLP)

## 1. Objetivo do artefato

Este artefato registra a implementacao de um baseline de Deep Learning com rede neural densa (MLP/Feedforward) para classificacao binaria de prospectividade mineral. O objetivo principal e estabelecer uma referencia quantitativa confiavel para a comparacao com modelos convolucionais (CNN) nas proximas etapas do projeto. Em vez de tratar o baseline como solucao final, a proposta aqui e construir um experimento reproduzivel, com justificativas metodologicas claras e analise critica conectada ao que foi observado no A1.

## 2. Escopo da entrega

A entrega esta concentrada na pasta `artefatos/a03_mlp_baseline/`, contendo o notebook `a03_mlp_baseline.ipynb` e esta documentacao. Durante a execucao, o notebook salva os resultados em `outputs/a03_mlp_baseline/`, incluindo modelo treinado, metricas em JSON, curvas de treino, matriz de confusao, curva ROC, tempos de execucao e arquivos de permutation importance. Essa organizacao foi mantida para facilitar auditoria, reexecucao e comparacao entre sprints.

## 3. Relacao com o A1 e fundamentacao

O A1 mostrou duas evidencias importantes para esta etapa: primeiro, que existe plausibilidade fisica para separacao espectral entre classes; segundo, que o pipeline de dados precisa controlar vazamento entre amostras correlacionadas. Por isso, o A3 reaproveita o mesmo contrato de rotulagem e os utilitarios do projeto, mantendo continuidade tecnica entre EDA e modelagem.

Tambem foi importante explicitar a dimensionalidade original. Cada amostra vem de um chip ASTER de 128x128 pixels com 9 bandas, o que gera `9 x 128 x 128 = 147456` colunas `pixel_*` no dataset tabular. Esse volume de features, combinado com amostragem relativamente pequena, aumenta risco de sobreajuste em MLP densa. A estrategia adotada foi reduzir dimensionalidade de forma controlada (agregacao por banda, normalizacao e PCA), sem perder totalmente o conteudo espectral.

## 4. Decisoes metodologicas

O split foi feito por `image_id` para evitar que amostras derivadas da mesma imagem aparecam em treino e teste, o que inflaria artificialmente desempenho. Em seguida, as features de pixel foram agregadas em medias por banda para tornar o baseline mais estavel neste regime de dados. A normalizacao por `StandardScaler` foi usada para estabilizar a otimizacao e o PCA com 95% de variancia explicada foi aplicado para reduzir redundancia e melhorar condicionamento numerico.

A arquitetura escolhida foi intencionalmente simples: duas camadas densas com ReLU, regularizacao L2, dropout e saida sigmoide para classificacao binaria. O treinamento foi conduzido com validacao explicita, early stopping e reducao adaptativa da taxa de aprendizado. Essa configuracao privilegia comparabilidade e consistencia de engenharia, em vez de tuning agressivo.

## 5. Reuso de codigo da base

Para manter padrao com o restante do repositorio, o notebook reaproveita funcoes de `src` para carregamento com grupos, normalizacao, PCA, ativacoes, metricas e visualizacao da matriz de confusao. Essa decisao reduz duplicacao de codigo e minimiza divergencia entre artefatos. A unica etapa implementada localmente foi a agregacao de `pixel_*` em medias por banda, pois depende diretamente da estrutura especifica deste experimento.

## 6. Avaliacao e leitura critica

A avaliacao quantitativa reporta accuracy, precision, recall, F1 e ROC-AUC, alem de matriz de confusao e curva ROC. Essa escolha segue a discussao do A1 de que acuracia isolada pode esconder comportamento indesejado em classificacao binaria aplicada a priorizacao de alvos.

A analise textual do notebook foi escrita para ser interpretativa e nao apenas descritiva. Os erros (FP/FN) sao discutidos como trade-off operacional, as curvas de treino/validacao sao usadas para diagnosticar estabilidade e o ranking de permutation importance e lido com cautela, como dependencia do modelo no experimento atual, e nao como causalidade geologica definitiva.

## 7. Reproducao

Para reproduzir, basta abrir o notebook e executar as celulas em ordem. O experimento usa `SEED = 42` para `random`, `numpy` e `tensorflow`, o que reduz variabilidade entre execucoes e melhora comparabilidade. Em ambiente novo, as dependencias podem ser instaladas com:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn jupyter
```

Depois disso:

```bash
jupyter notebook artefatos/a03_mlp_baseline/a03_mlp_baseline.ipynb
```

## 8. Limitacoes e proximos passos

Este baseline ainda tem limitacoes importantes. A representacao por medias de banda comprime o sinal espacial dos chips e pode ocultar estruturas locais relevantes para a geologia. Alem disso, o tamanho da amostra ainda restringe a capacidade de generalizacao de redes densas. Portanto, os resultados devem ser entendidos como linha de base.

Como continuidade, o projeto deve evoluir para CNN sob o mesmo protocolo experimental e incorporar testes de robustez mais fortes, como repeticao com seeds diferentes, curvas de aprendizado, avaliacao estratificada por segmentos geologicos e auditoria de similaridade entre imagens de treino e teste.
