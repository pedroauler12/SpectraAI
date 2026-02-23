## 3. Metodologia

### 3.1. Aquisição de Dados e Alvos Geológicos

A base de dados do projeto é composta por amostras de solo e rocha coletadas in situ pela **Frontera Minerals**, contendo teores geoquímicos de Elementos de Terras Raras (ETR). As amostras foram rotuladas binariamente:

* **Classe Positiva (y = 1):** Áreas com teores acima do *cut-off* econômico, associadas a depósitos iônicos ou rochas alcalinas mineralizadas.
* **Classe Negativa (y = 0):** Áreas estéreis ou com teores de base (background).

As assinaturas espectrais foram extraídas de imagens do sensor **ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer)**, utilizando as bandas do visível e infravermelho (VNIR) e infravermelho de ondas curtas (SWIR), com resolução espacial reamostrada para compatibilidade.

### 3.2. Pré-processamento e Engenharia de Atributos

Para mitigar ruídos e isolar a resposta mineralógica, o pipeline de dados executou:

1. **Filtragem de Máscaras:** Remoção de pixels contaminados por nuvens e densa cobertura vegetal (NDVI > limiar).
2. **Cálculo de Índices Minerais:** Foram geradas *features* baseadas em razões de bandas consagradas na literatura de sensoriamento remoto mineral, como o **Índice de Argilas** $[B06 / (B05 + B04)]$, visando destacar produtos de alteração hidrotermal e intemperismo.
3. **Vetorização:** Cada amostra foi convertida em um vetor de alta dimensionalidade ($p = 147.456$), representando tanto as bandas brutas quanto as janelas espaciais adjacentes.

### 3.3. Protocolo de Divisão de Dados (Anti-Leakage)

Um ponto crítico da metodologia é o controle de **vazamento de dados (spatial leakage)**. A divisão do dataset em treino (60%), validação (20%) e teste (20%) foi realizada no nível de cena (**image_id**).

* Amostras pertencentes à mesma imagem de satélite foram mantidas obrigatoriamente no mesmo grupo.
* Utilizou-se o método **StratifiedGroupKFold** para garantir que a proporção de classes fosse mantida em todos os *folds*, impedindo que o modelo memorizasse condições de iluminação ou sensores específicos de uma única imagem.

### 3.4. Modelagem Clássica (Baseline)

Foram avaliados três algoritmos de aprendizado supervisionado para estabelecer o desempenho de referência:

1. **Support Vector Machine (SVM):** Implementada com kernel linear e regularização , visando a maximização da margem em espaço de alta dimensionalidade.
2. **Random Forest (RF):** Conjunto de árvores de decisão para capturar interações não lineares entre as bandas espectrais.
3. **Regressão Logística:** Baseline linear para verificação de separabilidade simples e calibração probabilística.

A otimização de hiperparâmetros foi realizada via **GridSearchCV**, utilizando o conjunto de validação para a escolha final dos modelos.

### 3.5. Protocolo de Avaliação e Calibração de Limiar

Dada a natureza exploratória do problema, o limiar de decisão ($\tau$) não foi fixado em 0.5. Em vez disso:

1. O modelo gerou scores contínuos.
2. O limiar ótimo foi selecionado no conjunto de validação através da maximização do **F1-Score** na curva Precision-Recall.
3. As métricas finais (F1, Precision, Recall, ROC-AUC e PR-AUC) foram calculadas exclusivamente no conjunto de teste isolado.

$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$