# SpectraAI: Prospecção de Terras Raras a partir de Imagens Multiespectrais ASTER com Aprendizado de Máquina e Visão Computacional

### Autores: Drielly Santana Farias, Eduardo Farias Rizk, Giovanna Fátima de Britto Vieira, Larissa Martins Pereira de Souza, Lucas Ramenzoni Jorge,  Mateus Beppler Pereira, Pedro Auler de Barros Martins

## 1. Introdução

&emsp;&emsp; Os Elementos Terras Raras (Rare Earth Elements — REE) compõem um grupo de 17 elementos amplamente empregados em tecnologias de alto valor agregado, incluindo eletrônica, aplicações industriais avançadas e sistemas energéticos de baixo carbono. A demanda crescente por esses elementos, impulsionada pela transição energética global, tem intensificado preocupações quanto à segurança de suprimento, dado que a produção e o refino permanecem geograficamente concentrados (IEA, 2021; UNITED STATES GEOLOGICAL SURVEY, 2025). Nesse cenário, o Brasil ocupa posição estratégica em virtude de suas reservas expressivas de ETR, o que reforça a relevância de métodos eficientes e escaláveis de prospecção mineral no contexto nacional.

&emsp;&emsp; Do ponto de vista operacional, a prospecção mineral tradicional depende de campanhas de campo, amostragem e análises laboratoriais, etapas onerosas e de difícil escalabilidade espacial. Em contrapartida, o sensoriamento remoto oferece um meio de observação sistemática e repetível para apoiar a triagem de alvos, especialmente quando combinado a métodos quantitativos de análise de dados (SABINS, 1999; VAN DER MEER et al., 2012). Em particular, a exploração mineral por sensoriamento remoto se beneficia da relação entre resposta espectral e mineralogia/alteração, permitindo inferências indiretas sobre litologias e processos geológicos associados a mineralizações.

&emsp;&emsp; Nesse contexto, o Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) consolidou-se como um dos sensores mais utilizados em mapeamento litológico e exploração mineral por disponibilizar bandas espectrais relevantes em VNIR e SWIR, além de histórico robusto de aplicações documentadas na literatura (ABRAMS; YAMAGUCHI, 2019; RAMSEY; FLYNN, 2020). No entanto, a interpretação manual de cenas multiespectrais permanece limitada pela alta dimensionalidade espectral, heterogeneidade espacial e pela sutileza de padrões associados a mineralizações, o que pode introduzir subjetividade e restringir a reprodutibilidade dos resultados.

&emsp;&emsp;Historicamente, grande parte das aplicações de aprendizado de máquina em sensoriamento remoto mineral tem sido conduzida a partir de representações tabulares dos dados espectrais, nas quais cada pixel ou amostra é descrito como um vetor de atributos derivados das bandas disponíveis (SHIRMARD et al., 2022). Nesse contexto, modelos clássicos e redes neurais rasas, como Multi-Layer Perceptrons (MLP), foram amplamente empregados para tarefas de classificação litológica ou identificação de assinaturas minerais, sobretudo por sua capacidade de modelar relações não lineares entre variáveis espectrais. Entretanto, essa abordagem apresenta uma limitação importante: ao tratar cada amostra como um vetor independente, a estrutura espacial presente nas imagens multiespectrais é, em grande parte, descartada. Em problemas geológicos, essa informação espacial pode ser relevante, uma vez que processos de alteração mineral, zonas de contato litológico e padrões geomorfológicos tendem a se manifestar como estruturas contínuas ou texturas distribuídas no espaço.

&emsp;&emsp;Diante dessa limitação, abordagens baseadas em visão computacional têm sido cada vez mais exploradas em dados de sensoriamento remoto (ZHU et al., 2017). Em particular, redes neurais convolucionais (Convolutional Neural Networks — CNN) permitem aprender automaticamente padrões espaciais e espectrais a partir de janelas de imagem, preservando relações de vizinhança entre pixels e capturando estruturas que dificilmente seriam representadas por atributos tabulares isolados. No contexto de prospecção mineral, essa capacidade pode contribuir para identificar padrões sutis associados a zonas de alteração ou assinaturas espectrais distribuídas espacialmente, potencialmente ampliando a capacidade de generalização dos modelos.

&emsp;&emsp;Diante disso, este trabalho apresenta uma proposta metodológica para construção de um pipeline de ciência de dados geoespaciais, utilizando imagens ASTER e dados de referência fornecidos pela Frontera Minerals, com o objetivo de transformar as cenas em um conjunto supervisionado de amostras rotuladas e avaliar modelos de aprendizado de máquina e visão computacional para estimar, de forma probabilística, o potencial prospectivo em áreas de interesse. A hipótese central é que a incorporação de informação espacial por meio de representações em forma de chips multiespectrais possa permitir a extração automática de características relevantes, fornecendo uma base mais adequada para modelagem preditiva em tarefas de mapeamento prospectivo de ETR. A proposta privilegia a reprodutibilidade do processamento e a geração de evidências quantitativas que possam apoiar, em ciclos posteriores, validação geológica e refinamento do método.

&emsp;&emsp;As principais contribuições deste trabalho são: (i) a construção de um pipeline reprodutível de ciência de dados geoespaciais, desde a aquisição e rotulagem de cenas ASTER até a geração de chips multiespectrais supervisionados; (ii) a comparação sistemática entre modelos tabulares (baselines clássicos e MLP) e abordagens de visão computacional (CNN) para classificação de potencial prospectivo; e (iii) a avaliação do impacto de decisões arquiteturais e de regularização por meio de um estudo de ablação controlado.

#### 1.1 Objetivos da Pesquisa
##### Objetivo Geral

&emsp;&emsp;Desenvolver e avaliar um modelo de Deep Learning aplicado à visão computacional capaz de analisar imagens multiespectrais do sensor ASTER e estimar, de forma probabilística, o potencial de ocorrência de elementos de Terras Raras (REE), produzindo um ranking de prospectividade mineral que auxilie a priorização de áreas para campanhas de pesquisa geológica.

##### Objetivos Específicos

A partir desse objetivo geral, delineiam-se os seguintes objetivos específicos:

1. Realizar engenharia de atributos espectrais, investigando transformações e combinações de bandas do ASTER capazes de destacar assinaturas espectrais relacionadas a minerais de alteração hidrotermal (argilas, óxidos e carbonatos), frequentemente associados a sistemas mineralizados.

2. Desenvolver e treinar modelos CNN, com foco em arquiteturas de visão computacional capazes de extrair padrões espaciais e espectrais presentes nas imagens multiespectrais.

3. Avaliar o desempenho dos modelos treinados utilizando áreas com ocorrências conhecidas de Terras Raras, por meio das métricas F1-score e AUC-ROC.

4. Produzir um ranking de áreas com maior potencial prospectivo, permitindo priorizar regiões para futuras etapas de pesquisa mineral e campanhas de campo.

## 2. Fundamentação Teórica

&emsp;&emsp; A análise por sensoriamento remoto em geociências fundamenta-se na interação entre radiação eletromagnética e materiais geológicos, em que minerais e rochas exibem respostas espectrais condicionadas por composição e estrutura físico-química. Em aplicações de exploração mineral, técnicas clássicas incluem manipulações espectrais (por exemplo, razões de bandas) e transformações estatísticas (como Análise de Componentes Principais — PCA), frequentemente empregadas para realçar assinaturas diagnósticas de minerais de alteração e discriminar unidades litológicas (ABRAMS; YAMAGUCHI, 2019; ROWAN; MARS, 2003).

&emsp;&emsp; O sensor ASTER (a bordo do satélite Terra) foi projetado com subsistemas em VNIR e SWIR, cuja combinação favorece a investigação de características mineralógicas relevantes. A literatura descreve o VNIR com três canais (15 m) e o SWIR com seis canais (30 m), originalmente operacionais até a falha do subsistema SWIR em 2008, fato que impõe restrições temporais importantes para estudos baseados nessas bandas (ABRAMS; YAMAGUCHI, 2019; RAMSEY; FLYNN, 2020). Revisões abrangentes destacam que o ASTER contribuiu de forma significativa para mapeamento litológico e exploração mineral ao longo de décadas, consolidando práticas de processamento e interpretação em diferentes contextos geológicos (ABRAMS; YAMAGUCHI, 2019).

&emsp;&emsp; Estudos pioneiros demonstraram, de maneira empírica, a capacidade do ASTER em separar grupos litológicos e mapear contatos geológicos por meio de combinações de bandas e técnicas de realce espectral. Um exemplo clássico é o trabalho de Rowan e Mars (2003) na área de Mountain Pass (Califórnia), no qual dados ASTER foram empregados para mapear unidades litológicas e discriminar grupos relevantes para exploração mineral em condições adequadas de exposição superficial (ROWAN; MARS, 2003). Esses resultados, entretanto, dependem criticamente de escolhas de pré-processamento, seleção de atributos e critérios de avaliação, especialmente quando há variabilidade de iluminação, cobertura superficial e ruído residual.

&emsp;&emsp; Nos últimos anos, a integração de aprendizado de máquina tem ampliado o escopo do sensoriamento remoto aplicado à exploração mineral, ao permitir modelar relações não lineares e reduzir dependência de regras fixas (por exemplo, limiares definidos manualmente). Trabalhos recentes mostram pipelines supervisionados capazes de transformar atributos espectrais derivados de ASTER em classificações quantitativas, comparando algoritmos clássicos e redes neurais sob métricas objetivas (BAHRAMI et al., 2024). Em paralelo, revisões específicas sobre *mineral prospectivity mapping* com *deep learning* indicam crescimento do uso de modelos profundos, mas também ressaltam desafios recorrentes, como qualidade/escassez de rótulos, desbalanceamento de classes, ajustes de parâmetros, procedimentos de validação e robustez na generalização espacial (SUN et al., 2024).

&emsp;&emsp; Assim, a fundamentação deste projeto apoia-se em três pilares: (i) a adequação do ASTER e de suas bandas VNIR/SWIR para caracterização espectral relevante em mapeamento litológico e exploração mineral, amplamente documentada; (ii) a necessidade de procedimentos reprodutíveis de pré-processamento e construção de datasets para reduzir vieses e instabilidades; e (iii) a pertinência de métodos supervisionados (clássicos e profundos) para produzir escores probabilísticos e ranqueamento de alvos, com avaliação baseada em métricas que reflitam tanto desempenho quanto generalização espacial.

## 3. Materiais e Métodos

### 3.1 Materiais

&emsp;&emsp; Os materiais utilizados neste projeto reúnem dados orbitais, dados de referência geográfica e metadados de rotulagem fornecidos pela Frontera Minerals. A base orbital é composta por cenas ASTER do produto L2 Surface Reflectance VNIR and Crosstalk-Corrected SWIR (AST_07XT), consultadas no CMR/NASA Earthdata por `concept_id`. A janela temporal de coleta foi definida entre 2000 e 2007, recorte adotado para preservar a disponibilidade operacional das bandas SWIR no período de interesse.

&emsp;&emsp; Para análise espectral nesta etapa, foram priorizadas as bandas VNIR (B01, B02, B03N) e SWIR (B04-B09), totalizando nove bandas por chip multiespectral. As cenas candidatas são recuperadas por caixa geográfica ao redor dos pontos-alvo e, quando há mais de um granule elegível, aplica-se seleção por menor cobertura de nuvens registrada nos metadados. Assim, o critério de qualidade atmosférica empregado é de minimização relativa de nuvens, e não de exclusão absoluta.

&emsp;&emsp; O material de referência para supervisão é composto por coordenadas georreferenciadas de interesse geológico (incluindo Serra Verde e CBMM) e por listas de códigos positivos e negativos fornecidas pelo parceiro. Esses insumos orientam a associação entre amostras e rótulos no dataset final, constituindo o *ground truth* operacional utilizado nos experimentos. Após a remoção de amostras com rótulos inválidos, o dataset final é composto por 295 chips multiespectrais, dos quais 179 pertencem à classe negativa (60,7%) e 116 à classe positiva (39,3%), configurando um desbalanceamento moderado entre classes.

### 3.2 Métodos

&emsp;&emsp;A Figura 1 apresenta uma visão geral do pipeline metodológico adotado neste trabalho, desde a aquisição das cenas ASTER até a avaliação dos modelos de classificação.

Figura 1 – Pipeline metodológico do SpectraAI

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Aquisição ASTER│     │ Pré-processamento│     │  Geração de Chips   │
│  (VNIR + SWIR)  │────▶│ Filtragem, NDVI, │────▶│  128×128×9 pixels   │
│  2000–2007      │     │ Reprojeção WGS84 │     │  + Rotulagem binária│
└─────────────────┘     └──────────────────┘     └────────┬────────────┘
                                                          │
                                              ┌───────────┴───────────┐
                                              │                       │
                                              ▼                       ▼
                                    ┌──────────────────┐   ┌──────────────────┐
                                    │ Vetorização       │   │ Tensores 3D      │
                                    │ (tabular)         │   │ (espacial)       │
                                    └────────┬─────────┘   └────────┬─────────┘
                                             │                      │
                                             ▼                      ▼
                                    ┌──────────────────┐   ┌──────────────────┐
                                    │ Baselines:        │   │ CNN:             │
                                    │ SVM, RF, LR, MLP  │   │ Conv2D + Dense   │
                                    └────────┬─────────┘   └────────┬─────────┘
                                             │                      │
                                             └──────────┬───────────┘
                                                        ▼
                                              ┌──────────────────┐
                                              │   Avaliação      │
                                              │ F1-score, AUC-ROC│
                                              └──────────────────┘
```

#### 3.2.1. Aquisição de Dados e Alvos Geológicos

A base de dados do projeto é composta por amostras de solo e rocha coletadas *in situ* pela **Frontera Minerals**, contendo teores geoquímicos de Elementos de Terras Raras (ETR). As amostras foram rotuladas binariamente:

* **Classe Positiva (y = 1):** Áreas com teores acima do *cut-off* econômico, associadas a depósitos iônicos ou rochas alcalinas mineralizadas.
* **Classe Negativa (y = 0):** Áreas estéreis ou com teores de base (*background*).

As assinaturas espectrais foram extraídas de imagens do sensor **ASTER**, utilizando as bandas do visível e infravermelho (VNIR) e infravermelho de ondas curtas (SWIR). Devido à degradação do sensor SWIR após 2008, o pipeline prioriza cenas históricas (2000-2007) com cobertura de nuvens inferior a 20% (conforme documentado no protocolo de acesso ASTER), garantindo a integridade dos dados para mapeamento de argilas.

#### 3.2.2. Pré-processamento e Engenharia de Atributos

Para mitigar ruídos e isolar a resposta mineralógica, o pipeline executou:

1. **Filtragem de Máscaras:** Remoção de pixels contaminados por nuvens e densa cobertura vegetal (NDVI > limiar).
2. **Reprojeção:** Conversão sistemática de coordenadas para WGS84, corrigindo discrepâncias entre os dados de campo (SAD69) e os produtos orbitais.
3. **Cálculo de Índices Minerais:** Geração de *features* baseadas em razões de bandas como o **Índice de Argilas** $[B06 / (B05 + B04)]$.
4. **Vetorização (Abordagem Tabular):** Para a fase inicial de baselines, cada amostra foi convertida em um vetor de alta dimensionalidade ($p = 147.456$), representando bandas brutas e janelas adjacentes.

#### 3.2.3. Geração de Amostras (Chips) e Rotulagem

O dataset é construído a partir de **chips** gerados ao redor de pontos georreferenciados. Cada chip é um GeoTIFF multibanda com bandas VNIR+SWIR alinhadas.

* **Extração:** O recorte utiliza uma *bounding box* (bbox) com **jitter controlado por semente**, garantindo que o ponto de referência permaneça dentro do chip, mas em posições variadas para aumentar a robustez.
* **Processamento Tabular:** Para os modelos iniciais, os chips são convertidos em vetores (`pixel_*`) e metadados. A rotulagem é aplicada via mapeamento de `image_id` para as listas de positivos/negativos fornecidas no arquivo `extracted_codes.json`.

#### 3.2.4. Modelagem de Referência (Baselines)

Estabeleceu-se o desempenho de referência através de:

1. **Algoritmos Clássicos:** SVM (kernel linear), Random Forest e Regressão Logística, otimizados via `GridSearchCV`.
2. **Multi-Layer Perceptron (MLP):** Uma rede neural densa utilizada para testar a capacidade de aprendizado não linear sobre os dados vetorizados, servindo como o baseline de Deep Learning.

#### 3.2.5. Evolução para Visão Computacional (CNN)

Visando superar a limitação estrutural da MLP, que ignora a vizinhança espacial, o projeto evoluiu para o uso de **Redes Neurais Convolucionais (CNN)**:

* **Tensores Espaciais:** Em vez de vetorizar os pixels, a CNN recebe o chip em sua forma original (Altura x Largura x Canais), permitindo que filtros convolucionais identifiquem texturas e padrões morfológicos do solo.
* **Data Augmentation:** Implementação de rotações e espelhamentos para simular diferentes orientações geológicas e prevenir o *overfitting*.

#### 3.2.6 Protocolo de Divisão de Dados e Controle de Vazamento

&emsp;&emsp;O controle de vazamento de dados (data leakage) foi considerado na etapa de preparação dos conjuntos de treinamento, validação e teste. Inicialmente, o dataset tabular contendo os pixels extraídos das cenas ASTER é filtrado para remover amostras com rótulos inválidos. Em seguida, as amostras válidas são divididas por meio de amostragem estratificada, garantindo que a proporção entre as classes seja preservada em todos os subconjuntos.

&emsp;&emsp;A divisão é realizada em duas etapas utilizando a função train_test_split da biblioteca scikit-learn. Na primeira etapa, os dados são separados em treinamento (70%) e conjunto temporário (30%), mantendo a estratificação das classes. Na segunda etapa, o conjunto temporário é novamente dividido de forma estratificada em validação (15%) e teste (15%). Esse procedimento assegura que cada subconjunto represente adequadamente a distribuição original das classes.

&emsp;&emsp;Durante a preparação dos dados para a CNN, as amostras são convertidas em tensores com formato (N, H, W, C), compatível com o padrão channels-last utilizado pelo TensorFlow/Keras. Cada chip multiespectral é representado como um tensor 128 × 128 × 9, correspondendo às nove bandas espectrais selecionadas do sensor ASTER.

&emsp;&emsp;A normalização dos dados é realizada por padronização z-score por canal espectral, cujos parâmetros (média e desvio padrão) são estimados exclusivamente a partir do conjunto de treinamento. Esses mesmos parâmetros são posteriormente aplicados aos conjuntos de validação e teste, evitando vazamento de informação estatística entre os subconjuntos.

#### 3.2.7 Arquitetura da CNN e Hiperparâmetros

&emsp;&emsp;Para a etapa de visão computacional foi implementada uma rede neural convolucional como arquitetura baseline para experimentos com chips multiespectrais do sensor ASTER. A rede recebe tensores tridimensionais correspondentes aos chips extraídos das cenas, preservando a estrutura espacial e a informação espectral das bandas. Cada amostra possui dimensão 128 × 128 × 9, representando nove bandas espectrais selecionadas.

&emsp;&emsp;A arquitetura é composta por dois blocos convolucionais seguidos por camadas densas de classificação. Cada bloco inclui uma camada Conv2D com ativação ReLU, regularização L2 aplicada aos pesos e uma operação de MaxPooling2D para redução da dimensionalidade espacial. Após a extração de características, o tensor é convertido em vetor por meio da operação Flatten e processado por uma camada densa com 128 unidades. A camada de saída utiliza ativação sigmoid para produzir a probabilidade associada à classe positiva.

&emsp;&emsp;Para investigar o impacto de escolhas arquiteturais e de treinamento, foram definidas duas configurações experimentais utilizadas no estudo de ablação: uma configuração baseline e uma configuração com maior regularização por meio de taxas de dropout mais elevadas e learning rate reduzido. A Tabela 1 apresenta os principais hiperparâmetros utilizados nas duas configurações, destacando que apenas as taxas de dropout nas camadas convolucionais e densas, bem como o learning rate do otimizador, foram alterados entre os experimentos, enquanto os demais parâmetros foram mantidos constantes para permitir comparação controlada entre as variantes do modelo.

Tabela 1 – Hiperparâmetros utilizados nas configurações do ablation study

| Parâmetro | baseline | higher_dropout | deeper_network | smaller_input | higher_lr_only | higher_dropout_only |
|---|---|---|---|---|---|---|
| input_shape | [128,128,9] | [128,128,9] | [128,128,9] | [64,64,9] | [128,128,9] | [128,128,9] |
| num_classes | 2 | 2 | 2 | 2 | 2 | 2 |
| filters | [32,64] | [32,64] | [32,64,128] | [32,64] | [32,64] | [32,64] |
| kernel_size | 3 | 3 | 3 | 3 | 3 | 3 |
| pool_size | 2 | 2 | 2 | 2 | 2 | 2 |
| l2_regularizer | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| conv_dropout_rate | 0.2 | 0.3 | 0.2 | 0.2 | 0.2 | 0.3 |
| dense_dropout_rate | 0.5 | 0.6 | 0.5 | 0.5 | 0.5 | 0.6 |
| dense_units | 128 | 128 | 128 | 128 | 128 | 128 |
| batch_size | 32 | 32 | 32 | 32 | 32 | 32 |
| epochs | 50 | 50 | 50 | 50 | 50 | 50 |
| learning_rate | 0.001 | 0.0005 | 0.001 | 0.001 | 0.0005 | 0.001 |
| optimizer | adam | adam | adam | adam | adam | adam |
| objetivo do teste | configuração base | maior regularização | maior profundidade | reduzir custo computacional | isolar efeito do LR | isolar efeito do dropout |
---

&emsp;&emsp;O protocolo experimental foi estruturado para avaliar a capacidade de redes neurais convolucionais em identificar padrões associados à presença de elementos de terras raras a partir de dados multiespectrais ASTER. O conjunto de dados foi dividido em três subconjuntos independentes: treinamento (70%), validação (15%) e teste (15%), utilizando amostragem estratificada para preservar a proporção de classes. Durante o treinamento, o conjunto de validação é utilizado para monitorar o desempenho da rede ao longo das épocas e identificar possíveis sinais de sobreajuste (*overfitting*), enquanto o conjunto de teste permanece isolado e é utilizado apenas na avaliação final do modelo selecionado. O desempenho é avaliado principalmente por meio das métricas F1-score e área sob a curva ROC (ROC-AUC), adequadas para cenários com possível desbalanceamento entre classes.

## 4. Trabalhos Relacionados

#### 4.1 Twenty Years of ASTER Contributions to Lithologic Mapping and Mineral Exploration

&emsp;&emsp; O artigo de revisão "Twenty Years of ASTER Contributions to Lithologic Mapping and Mineral Exploration", publicado por Abrams e Yamaguchi (2019), resume o histórico de aplicações bem-sucedidas do sensor ASTER na pesquisa e mapeamento mineral. Lançado em 1999, o ASTER revolucionou a exploração geológica global ao fornecer melhor resolução espacial e capacidades multiespectrais únicas, apresentando seis bandas no infravermelho de ondas curtas (SWIR) e cinco bandas no infravermelho termal (TIR).

&emsp;&emsp; Essa configuração espectral superou as limitações de satélites anteriores, como o Landsat, permitindo a distinção precisa de grupos minerais diagnósticos de alteração hidrotermal — como argilas, carbonatos, sulfatos e distinções na composição de silicatos. No contexto geológico voltado para minerais críticos e de Terras Raras, a revisão de Abrams e Yamaguchi destaca trabalhos pioneiros, como o estudo de Rowan e Mars (2003), que foram os primeiros a demonstrar a capacidade das 14 bandas do ASTER em distinguir litologias e mapear zonas de contato metamórfico associadas a depósitos de minerais de terras raras na região de Mountain Pass, Califórnia.

&emsp;&emsp; A revisão literária também aborda a evolução das técnicas aplicadas ao extenso volume de imagens do ASTER para extração de informações mineralógicas:os autores relatam o uso bem-sucedido de técnicas mais simples, como índices minerais baseados em razões de bandas (band ratios), até métodos de processamento estatístico, como Análise de Componentes Principais (PCA). Ademais, o artigo relata o uso crescente de métodos analíticos sofisticados nos últimos anos, incluindo machine learning e modelos de redes neurais (como as redes neurais MLP e modelos SOM) utilizados para classificar complexidades espaciais e realizar mapeamentos litológicos e de zonas de alteração.

&emsp;&emsp; Essa trajetória documentada por Abrams e Yamaguchi (2019) corrobora o problema e a justificativa metodológica que escolhemos. O artigo confirma que as imagens ASTER possuem dados  suficientes para caracterizar as assinaturas espectrais associadas a depósitos minerais. No entanto, a alta dimensionalidade e a complexidade espacial desses dados tornam a análise manual desafiadora, especialmente para padrões sutis. Dessa forma, o histórico literário valida a criação do pipeline de ciência de dados e o uso de algoritmos de Deep Learning e Visão Computacional, atestando a viabilidade técnica de utilizar os dados multiespectrais ASTER como a principal fonte de evidências para estimar e rankear áreas prospectivas de forma mais objetiva, escalável e probabilística.

#### 4.2 Machine Learning-Based Lithological Mapping from ASTER Remote-Sensing Imagery

&emsp;&emsp; Um avanço recente e relevante é o estudo de Bahrami et al. (2024), que investiga mapeamento litológico automatizado a partir de imagens ASTER por meio de uma comparação sistemática entre algoritmos de machine learning tradicionais (Random Forest, SVM, Gradient Boosting e XGBoost) e uma abordagem de deep learning (ANN) aplicada ao caso da região mineralizada de Sar-Cheshmeh (Irã). O trabalho se destaca por estruturar um pipeline comparável ao de exploração mineral baseada em sensoriamento remoto, incorporando engenharia/seleção de atributos espectrais (features derivadas de bandas e análise de correlação/importance) e avaliando quantitativamente o desempenho dos modelos via acurácia global para diferentes classes litológicas.
&emsp;&emsp; Como contribuição para este projeto, Bahrami et al. reforçam que o ASTER mantém alta utilidade para tarefas de classificação litológica e identificação indireta de minerais quando combinado com métodos supervisionados, além de evidenciar que escolhas de pré-processamento e seleção de variáveis afetam significativamente a qualidade do mapa final (BAHRAMI et al., 2024).
&emsp;&emsp; Entretanto, há limitações importantes quando comparamos com a proposta da Frontera Minerals. Primeiro, o estudo é orientado a classes litológicas em um contexto regional específico, não sendo desenhado diretamente para um problema de “detecção/ranking prospectivo” (ex.: presença/ausência de assinatura associada a Terras Raras em torno de ocorrências conhecidas). Segundo, o trabalho depende de um conjunto de treinamento bem definido para classes do mapeamento local, enquanto o desafio do projeto envolve generalização e rotulagem positiva/negativa por proximidade geográfica (chips ao redor de coordenadas de referência), o que tende a introduzir ruído de rótulo e exigir estratégias de validação e modelagem. Ainda assim, o artigo oferece um baseline metodológico sólido para justificar a etapa de comparação entre modelos clássicos e redes neurais usando ASTER, além de servir de referência para decisões de features e avaliação.

#### 4.3 Redes Neurais para Prospecção de Terras Raras

&emsp;&emsp;Avançando além da caracterização espectral dos sensores, a integração de modelos baseados em aprendizado profundo (_Deep Learning_) surge como o passo evolutivo necessário para superar a sutileza das assinaturas de elementos de terras raras (REE). O trabalho de Luo et al. (2025) introduziu o framework **DEEP-SEAM v1.0**, demonstrando que a natureza não linear e altamente heterogênea dos conjuntos de dados de exploração impõe limitações aos métodos tradicionais de mapeamento.

&emsp;&emsp;Para solucionar essa complexidade, os autores empregam redes neurais para extrair padrões ocultos em dados multifonte, aplicando a Deviation Network (DevNet) para identificar anomalias mesmo em cenários de dados esparsos e desbalanceados. Complementando essa visão técnica, o estudo consolidado de Song et al. (2024) reforça que redes multitarefa podem filtrar ruídos e descobrir correlações não lineares entre composições químicas e a presença de REEs, reduzindo gargalos de custo e tempo em relação às análises estatísticas convencionais.

&emsp;&emsp;Essa convergência entre modelos _data-driven_ e a necessidade de interpretar assinaturas minerais complexas corrobora a adoção de redes neurais no SpectraAI. Ao utilizar redes neurais e visão computacional para processar imagens ASTER, o projeto promove o ranqueamento de áreas prospectivas de terras raras de forma escalável, objetiva e com alta fidelidade geológica.

#### 4.4 Machine Learning em Sensoriamento Remoto para Exploração Mineral

&emsp;&emsp;Em uma revisão abrangente, Shirmard et al. (2022) catalogam e comparam o uso de técnicas de machine learning aplicadas a dados de sensoriamento remoto para exploração mineral, incluindo métodos baseados em pixels (tabulares) e abordagens baseadas em patches espaciais. Os autores analisam diferentes sensores — entre eles ASTER, Landsat e Sentinel — e identificam que poucos estudos realizam comparações sistemáticas entre representações tabulares e espaciais sobre o mesmo dataset e com as mesmas métricas de avaliação. Essa lacuna é diretamente endereçada pelo SpectraAI, que implementa um pipeline unificado comparando baselines clássicos (SVM, Random Forest, Regressão Logística), MLP e CNN sob F1-score e AUC-ROC no mesmo conjunto de dados, permitindo uma avaliação direta do ganho obtido pela incorporação de informação espacial.

#### 4.5 Deep Learning em Dados Hiperespectrais

&emsp;&emsp;O trabalho pioneiro de Chen et al. (2014) foi um dos primeiros a aplicar deep learning à classificação de dados hiperespectrais de sensoriamento remoto. Utilizando stacked autoencoders e redes convolucionais, os autores demonstraram ganhos significativos sobre métodos clássicos como SVM, especialmente quando patches espaciais são empregados como forma de aumentar a quantidade efetiva de amostras de treinamento. A abordagem de usar recortes espaciais (patches) para capturar relações de vizinhança é análoga à estratégia de chips adotada pelo SpectraAI. Entretanto, Chen et al. trabalham com dados hiperespectrais em cenas de uso do solo, enquanto o SpectraAI aplica a abordagem a dados multiespectrais ASTER voltados especificamente para prospecção mineral de terras raras, um problema com desbalanceamento de classes e rótulos derivados de referência geológica.

#### 4.6 Síntese Comparativa

&emsp;&emsp;A Tabela 2 sintetiza os trabalhos relacionados discutidos, destacando o método empregado, os dados utilizados e a principal lacuna em relação ao SpectraAI.

Tabela 2 – Síntese comparativa dos trabalhos relacionados

| Trabalho | Método | Dados | Lacuna em relação ao SpectraAI |
|---|---|---|---|
| Abrams e Yamaguchi (2019) | Revisão (razões de bandas, PCA) | ASTER multibanda | Sem ML/DL automatizado |
| Bahrami et al. (2024) | RF, SVM, XGBoost, ANN | ASTER | Classificação litológica, não ranking prospectivo binário |
| Luo et al. (2025) | DevNet (semi-supervisionado) | Dados multifonte | Não usa sensoriamento remoto multiespectral nativo |
| Song et al. (2024) | Redes multitarefa | Dados geoquímicos | Dados tabulares de composição, sem imagens |
| Shirmard et al. (2022) | Revisão (ML + sensoriamento remoto) | Diversos sensores | Identifica lacuna de comparações tabular vs. espacial |
| Chen et al. (2014) | Autoencoders, CNN | Hiperespectral | Uso do solo, não prospecção mineral de REE |

## 5. Discussão

&emsp;&emsp;Os resultados obtidos indicam que a utilização de chips multiespectrais do sensor ASTER combinados com redes neurais convolucionais constitui uma abordagem promissora para a identificação de padrões associados à presença de elementos de terras raras. A estrutura espacial dos chips, aliada à informação espectral distribuída nas diferentes bandas, permite que o modelo aprenda representações discriminativas diretamente a partir dos dados. Dessa forma, a CNN atua como um mecanismo de extração automática de características capaz de capturar relações espaciais e espectrais relevantes, reduzindo a dependência de engenharia manual de atributos.

&emsp;&emsp;A organização do pipeline experimental buscou consistência metodológica e confiabilidade na avaliação do modelo. A divisão dos dados em conjuntos de treinamento, validação e teste contribui para preservar a distribuição das classes ao longo do processo de modelagem, enquanto o isolamento do conjunto de teste até a etapa final evita vieses na estimativa de desempenho. Nesse contexto, o uso do F1-score e da área sob a curva ROC (ROC-AUC) permite avaliar simultaneamente o equilíbrio entre precisão e recall e a capacidade discriminativa do modelo em diferentes limiares de decisão.

&emsp;&emsp;Os experimentos conduzidos no ablation study oferecem uma análise adicional sobre o impacto de decisões arquiteturais e de hiperparâmetros no comportamento do modelo. A comparação entre diferentes níveis de dropout e valores de learning rate permite observar como mecanismos de regularização influenciam a capacidade de generalização da rede. Em particular, a combinação de penalização L2 nas camadas convolucionais e camadas de dropout atua como um controle sobre a complexidade efetiva do modelo, reduzindo a tendência de memorização de padrões específicos do conjunto de treinamento, o que é especialmente relevante em cenários com dados geoespaciais limitados ou fortemente correlacionados.

&emsp;&emsp;Apesar dos resultados encorajadores, algumas limitações devem ser consideradas. A arquitetura empregada foi intencionalmente simples e utilizada como modelo de referência inicial, o que sugere a possibilidade de melhorias por meio de arquiteturas mais profundas ou estratégias adicionais de regularização e ajuste de hiperparâmetros. Trabalhos futuros podem explorar redes convolucionais mais complexas, diferentes resoluções espaciais dos chips e a integração de atributos geoespaciais derivados. Ainda assim, o pipeline desenvolvido demonstra o potencial do uso de aprendizado profundo aplicado a dados de sensoriamento remoto como ferramenta de apoio à prospecção mineral, permitindo ordenar áreas de interesse com base em escores probabilísticos de potencial prospectivo.

### Referências

**ABRAMS, M.; YAMAGUCHI, Y.** Twenty Years of ASTER Contributions to Lithologic Mapping and Mineral Exploration. *Remote Sensing*, v. 11, n. 11, art. 1394, 2019. DOI: 10.3390/rs11111394. Disponível em: [https://doi.org/10.3390/rs11111394](https://doi.org/10.3390/rs11111394). Acesso em: 22 fev. 2026.

**BAHRAMI, H. et al.** Machine Learning-Based Lithological Mapping from ASTER Remote-Sensing Imagery. *Minerals*, v. 14, n. 2, art. 202, 2024. DOI: 10.3390/min14020202. Disponível em: [https://doi.org/10.3390/min14020202](https://doi.org/10.3390/min14020202). Acesso em: 24 fev. 2026.

**CHEN, Y. et al.** Deep Learning-Based Classification of Hyperspectral Data. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, v. 7, n. 6, p. 2094–2107, 2014. DOI: 10.1109/JSTARS.2014.2329330. Disponível em: [https://doi.org/10.1109/JSTARS.2014.2329330](https://doi.org/10.1109/JSTARS.2014.2329330). Acesso em: 12 mar. 2026.

**INTERNATIONAL ENERGY AGENCY (IEA).** The Role of Critical Minerals in Clean Energy Transitions. Paris: IEA, 2021. Disponível em: [https://www.iea.org/reports/the-role-of-critical-minerals-in-clean-energy-transitions](https://www.iea.org/reports/the-role-of-critical-minerals-in-clean-energy-transitions). Acesso em: 18 mar. 2026.

**LUO, Z.** et al. An explainable semi-supervised deep learning framework for mineral prospectivity mapping: DEEP-SEAM v1.0. *EGUsphere* (preprint), 2025. DOI: 10.5194/egusphere-2025-3283. Disponível em: [https://doi.org/10.5194/egusphere-2025-3283](https://doi.org/10.5194/egusphere-2025-3283). Acesso em: 23 fev. 2026.

**NATIONAL AERONAUTICS AND SPACE ADMINISTRATION (NASA).** ASTER L2 Surface Reflectance VNIR and Crosstalk-Corrected SWIR (AST_07XT) — Product Description. *NASA Earthdata*, s.d. Disponível em: [https://earthdata.nasa.gov/](https://earthdata.nasa.gov/). Acesso em: 26 fev. 2026.

**RAMSEY, M. S.; FLYNN, I. T. W.** The Spatial and Spectral Resolution of ASTER Infrared Image Data: A Paradigm Shift in Volcanological Remote Sensing. *Remote Sensing*, v. 12, n. 4, art. 738, 2020. DOI: 10.3390/rs12040738. Disponível em: [https://doi.org/10.3390/rs12040738](https://doi.org/10.3390/rs12040738). Acesso em: 26 fev. 2026.

**ROWAN, L. C.; MARS, J. C.** Lithologic mapping in the Mountain Pass, California area using ASTER data. *Remote Sensing of Environment*, v. 84, n. 3, p. 350–366, 2003. DOI: 10.1016/S0034-4257(02)00127-X. Disponível em: [https://doi.org/10.1016/S0034-4257(02)00127-X](https://doi.org/10.1016/S0034-4257(02)00127-X). Acesso em: 26 fev. 2026.

**SABINS, F. F.** Remote sensing for mineral exploration. *Ore Geology Reviews*, v. 14, n. 3-4, p. 157–183, 1999. DOI: 10.1016/S0169-1368(99)00007-4. Disponível em: [https://doi.org/10.1016/S0169-1368(99)00007-4](https://doi.org/10.1016/S0169-1368(99)00007-4). Acesso em: 18 mar. 2026.

**SHIRMARD, H. et al.** A review of machine learning in processing remote sensing data for mineral exploration. *Remote Sensing of Environment*, v. 268, art. 112750, 2022. DOI: 10.1016/j.rse.2021.112750. Disponível em: [https://doi.org/10.1016/j.rse.2021.112750](https://doi.org/10.1016/j.rse.2021.112750). Acesso em: 12 mar. 2026.

**SONG, Y.** et al. Predicting rare earth elements concentration in coal ashes with multi-task neural networks. *Materials Horizons*, v. 11, p. 747–757, 2024. DOI: 10.1039/D3MH01491F. Disponível em: [https://doi.org/10.1039/D3MH01491F](https://doi.org/10.1039/D3MH01491F). Acesso em: 23 fev. 2026.

**SUN, K.** et al. A Review of Mineral Prospectivity Mapping Using Deep Learning. *Minerals*, v. 14, n. 10, art. 1021, 2024. DOI: 10.3390/min14101021. Disponível em: [https://doi.org/10.3390/min14101021](https://doi.org/10.3390/min14101021). Acesso em: 26 fev. 2026.

**UNITED STATES GEOLOGICAL SURVEY (USGS).** Interior Department releases final 2025 List of Critical Minerals. *U.S. Geological Survey*, 14 nov. 2025. Disponível em: [https://www.usgs.gov/news/science-snippet/interior-department-releases-final-2025-list-critical-minerals](https://www.usgs.gov/news/science-snippet/interior-department-releases-final-2025-list-critical-minerals). Acesso em: 26 fev. 2026.

**VAN DER MEER, F. D. et al.** Multi- and hyperspectral geologic remote sensing: A review. *International Journal of Applied Earth Observation and Geoinformation*, v. 14, n. 1, p. 112–128, 2012. DOI: 10.1016/j.jag.2011.08.002. Disponível em: [https://doi.org/10.1016/j.jag.2011.08.002](https://doi.org/10.1016/j.jag.2011.08.002). Acesso em: 18 mar. 2026.

**ZHU, X. X. et al.** Deep Learning in Remote Sensing: A Comprehensive Review and List of Resources. *IEEE Geoscience and Remote Sensing Magazine*, v. 5, n. 4, p. 8–36, 2017. DOI: 10.1109/MGRS.2017.2762307. Disponível em: [https://doi.org/10.1109/MGRS.2017.2762307](https://doi.org/10.1109/MGRS.2017.2762307). Acesso em: 18 mar. 2026.