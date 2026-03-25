# SpectraAI: Prospecção de Terras Raras a partir de Imagens Multiespectrais ASTER com Aprendizado de Máquina e Visão Computacional

### Autores: Drielly Santana Farias, Eduardo Farias Rizk, Giovanna Fátima de Britto Vieira, Larissa Martins Pereira de Souza, Lucas Ramenzoni Jorge,  Mateus Beppler Pereira, Pedro Auler de Barros Martins

## Resumo

A prospecção de Elementos de Terras Raras (REE) permanece desafiadora do ponto de vista da escalabilidade e reprodutibilidade, dependendo predominantemente de campanhas custosas de campo e análises laboratoriais subjetivas. Este trabalho apresenta um pipeline integrado de ciência de dados geoespaciais capaz de transformar imagens multiespectrais ASTER em evidências quantitativas e probabilísticas de potencial prospectivo de REE, contribuindo para automatização e redução de custos em etapas iniciais de prospecção. A metodologia começa pela aquisição rigorosa de cenas ASTER (2000-2007) com filtragem de qualidade atmosférica, seguida de pré-processamento geoespacial com protocolos de validação que reduzem data leakage e implementam engenharia de atributos espectrais. Em seguida, constrói-se chips multibanda (128×128×9) supervisionados e rotulados a partir de dados de referência geológica fornecidos pela Frontera Minerals. A avaliação comparativa abrange modelos clássicos (SVM, Random Forest, Regressão Logística), Deep Learning tabular (MLP) e visão computacional (CNN), seguindo rigoroso protocolo de ablação que isola o impacto real de variantes arquiteturais e hiperparâmetros sob condições controladas. Os resultados, obtidos mediante métricas adequadas ao desbalanceamento de classes (F1-score, acurácia balanceada, ROC-AUC) e estrutura de validação com isolamento robusto de conjunto de teste, mostram que o SVM alcança ROC-AUC ~0,88 e F1 > 0,85 em teste, validando a eficácia dos atributos espectrais extraídos. A CNN baseline, treinada em 236 amostras com validação e teste estratificados, atinge acurácia de validação ~0,90 (época 35) e F1 ponderado ~0,83, demonstrando potencial para capturar relações espaciais-espectrais em chips ASTER e suindicando vantagem competitivaobre abordagens tabulares em cenários com dados limitados. Embora os experimentos se encontrem em estágio ativo de iteração (N=295), a convergência metodológica entre seleção de atributos, protocolos geoespaciais rigorosos e modelagem deep learning aponta para viabilidade técnica e científica da proposta como ferramenta de triagem prospectiva auxiliada. Perspectivas futuras incluem validação geológica em campo, refinamento arquitetural em etapas posteriores e integração de dados multissensor para ampliar cobertura espectral e robustez de previsão.

**Palavras-chave:** sensoriamento remoto, ASTER, elementos de terras raras, aprendizado de máquina, redes neurais convolucionais, prospecção mineral, dados geoespaciais.

## 1. Introdução

&emsp;&emsp;Os Elementos Terras Raras (Rare Earth Elements — REE) compõem um grupo de 17 elementos com propriedades físico-químicas únicas, amplamente empregados em tecnologias estratégicas como baterias de veículos elétricos, turbinas eólicas, eletrônica de consumo e sistemas de defesa. A relevância econômica e geopolítica desses elementos tem sido reiterada por órgãos oficiais e relatórios setoriais recentes, que destacam vulnerabilidades em cadeias globais de suprimento e riscos associados à alta concentração geográfica de produção e refino (UNITED STATES GEOLOGICAL SURVEY, 2025). Nesse cenário, a identificação de novas jazidas em território nacional ganha relevância estratégica direta.

&emsp;&emsp;Do ponto de vista operacional, a prospecção mineral tradicional depende de campanhas de campo, amostragem e análises laboratoriais que são etapas onerosas, lentas e de difícil escalabilidade espacial. Em contrapartida, o sensoriamento remoto oferece um meio de observação sistemática e repetível para apoiar a triagem inicial de alvos, especialmente quando combinado a métodos quantitativos de análise de dados. A relação entre resposta espectral e mineralogia permite inferências indiretas sobre litologias e processos geológicos associados a mineralizações, tornando essa abordagem particularmente adequada para reduzir o escopo de campanhas de campo antes que recursos sejam mobilizados.

&emsp;&emsp;Nesse contexto, o Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) consolidou-se como um dos sensores mais utilizados em mapeamento litológico e exploração mineral, ao disponibilizar bandas espectrais relevantes no VNIR e no SWIR, com histórico robusto de aplicações documentadas na literatura (ABRAMS; YAMAGUCHI, 2019; RAMSEY; FLYNN, 2020). A janela temporal de dados SWIR operacionais, disponível entre 2000 e 2008 (antes da falha do subsistema), impõe restrições à aquisição, mas representa um acervo histórico de cobertura global ainda amplamente explorado. No entanto, a interpretação manual dessas cenas permanece limitada pela alta dimensionalidade espectral, heterogeneidade espacial e pela sutileza de padrões associados a mineralizações, introduzindo subjetividade e restringindo a reprodutibilidade dos resultados, conforme identificado em revisões sistemáticas da área (SHIRMARD et al., 2022).

&emsp;&emsp;Historicamente, abordagens de aprendizado de máquina em sensoriamento remoto mineral trataram os dados espectrais de forma tabular, descrevendo cada pixel como um vetor de atributos derivados das bandas disponíveis. Modelos como Random Forest e Support Vector Machines demonstraram capacidade de modelar relações não lineares entre variáveis espectrais (BAHRAMI et al., 2024), mas compartilham uma limitação estrutural: ao descartar a componente espacial, ignoram o fato de que processos de alteração mineral, zonas de contato litológico e padrões geomorfológicos se manifestam como estruturas contínuas no espaço. Redes neurais convolucionais (CNN) superam essa restrição ao preservar relações de vizinhança entre pixels e extrair automaticamente características espaciais e espectrais a partir de janelas de imagem, potencialmente ampliando a capacidade discriminativa em cenários com poucos dados rotulados.

&emsp;&emsp;É nesse contexto que se insere o presente trabalho, desenvolvido em parceria com a Frontera Minerals a partir de dados georreferenciados de ocorrências conhecidas de REE. A hipótese central é que a incorporação de informação espacial por meio de chips multiespectrais rotulados permite extrair características mais representativas do que abordagens puramente tabulares, resultando em modelos preditivos mais robustos para triagem prospectiva. O pipeline proposto transforma imagens ASTER em evidências quantitativas e probabilísticas de potencial prospectivo, contribuindo para a automatização e redução de custos nas etapas iniciais de exploração geológica.

&emsp;&emsp;O trabalho contribui por meio de três vertentes complementares: (i) estabelecimento de um pipeline reprodutível de ciência de dados geoespaciais, desde a aquisição de cenas ASTER até a geração supervisionada de chips multiespectrais (128×128×9), com protocolos rigorosos de controle de vazamento de dados; (ii) comparação sistemática entre modelos tabulares como baselines clássicos (SVM, Random Forest, Regressão Logística) alimentados por atributos espectrais extraídos e rede neural densa (MLP), e abordagem de visão computacional (CNN), avaliando qual representação melhor captura características espectrais-espaciais relevantes para prospecção; e (iii) estudo de ablação controlado para isolar o impacto de decisões arquiteturais e de regularização, aspecto especialmente crítico dado o tamanho reduzido do dataset disponível (N=295).

&emsp;&emsp;As seções seguintes apresentam a fundamentação teórica que sustenta as escolhas metodológicas, os materiais e métodos empregados, os resultados experimentais obtidos e uma conclusão preliminar com os principais achados e perspectivas de continuidade.

#### 1.1 Objetivos da Pesquisa

##### Objetivo Geral

&emsp;&emsp;Desenvolver e avaliar um modelo de Deep Learning aplicado à visão computacional capaz de analisar imagens multiespectrais do sensor ASTER e estimar, de forma probabilística, o potencial de ocorrência de Elementos de Terras Raras (REE), produzindo um ranking de prospectividade mineral que subsidie a priorização de áreas para campanhas de pesquisa geológica de campo.

##### Objetivos Específicos

&emsp;&emsp;Para verificar a hipótese central, estabelecem-se os seguintes objetivos específicos:

(i) Construir protocolo robusto de engenharia de atributos espectrais, investigando transformações e combinações de bandas ASTER para realce de minerais de alteração hidrotermal (argilas, óxidos e carbonatos), atributos que alimentam tanto os modelos tabulares quanto servem de referência interpretativa para os experimentos com CNN.

(ii) Desenvolver e comparar arquiteturas de visão computacional (CNN) com baselines supervisionados clássicos (SVM, Random Forest, Regressão Logística) e tabular (MLP), avaliando qual abordagem captura melhor as características espectrais-espaciais relevantes para prospecção de REE.

(iii) Implementar validação estratificada com isolamento rigoroso do conjunto de teste, utilizando métricas robustas ao desbalanceamento de classes (F1-score, acurácia balanceada, ROC-AUC) para avaliação confiável de desempenho e generalização.

(iv) Produzir ranking de áreas com maior potencial prospectivo, demonstrando a aplicabilidade prática do pipeline como ferramenta de apoio à tomada de decisão em exploração geológica e à priorização de campanhas futuras.

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
* **Data Augmentation:** Estratégia de aumento de dados aplicada exclusivamente ao conjunto de treinamento para expandir a diversidade de exemplos mantendo a integridade da avaliação. As transformações implementadas incluem: (i) `RandomFlip` com modo horizontal e vertical para simular diferentes perspectivas espaciais; (ii) `RandomRotation` com fator de 0,08 (±28,8°) para capturar variações de orientação geológica; (iii) `RandomContrast` com fator de 0,2 (redução de até 20% do contraste) para robustez a variações de iluminação e condições de aquisição. Uma análise sistemática explorou sete configurações de intensidade de augmentação (baseline, leve, moderada, intensa, rotação-aumentada, contraste-aumentado, combinada), com resultados documentados em `outputs/a08_transfer_learning/`, permitindo identificar a intensidade ótima de transformação que maximiza generalização sem prejudicar a representatividade dos padrões espectrais de interesse.

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

#### 3.2.8. Transfer Learning e Adaptação de Redes Pré-Treinadas

&emsp;&emsp;Reconhecendo o tamanho limitado do dataset (N=295) e a complexidade de treinar arquiteturas profundas do zero, o pipeline incorporou a estratégia de transfer learning por meio do backbone **MobileNetV2**, arquitetura eficiente pré-treinada em ImageNet. Essa abordagem aproveita representações de features genéricas aprendidas em vastos conjuntos de dados visuais e as adapta ao domínio específico de prospecção de terras raras a partir de dados multiespectrais ASTER.

&emsp;&emsp;**Adaptação Espectral (Camada de Conversão 1×1):** O sensor ASTER fornece 9 bandas espectrais (VNIR + SWIR), enquanto o MobileNetV2 pré-treinado em ImageNet espera 3 canais (RGB). Em vez de descartar bandas ou redimensionar arbitrariamente os dados, o pipeline implementa uma camada adaptadora de convolução 1×1, que realiza uma combinação linear das 9 bandas para projetá-las em um espaço latente de 3 canais. Essa estratégia preserva a riqueza de informação espectral enquanto viabiliza o carregamento dos pesos pré-treinados, adicionando apenas ~30 parâmetros treináveis na entrada do modelo.

&emsp;&emsp;**Estratégia de Treinamento em Duas Fases:**

(i) **Fase 1 — Head Training (4 epochs, Learning Rate 1e-4):** O backbone MobileNetV2 permanece totalmente congelado, preservando os pesos pré-treinados do ImageNet. Apenas a camada adaptadora espectral e a cabeça de classificação (camadas Dense) são treinadas. Callbacks de parada antecipada (`EarlyStopping`) e redução de taxa de aprendizado (`ReduceLROnPlateau`) monitoram `val_loss` para evitar overfitting. Essa fase adapta o modelo ao domínio ASTER sem degradar características pré-aprendidas.

(ii) **Fase 2 — Fine-Tuning Parcial (8 epochs):** Após o head estar adequadamente adaptado, desbloqueiam-se as últimas 20 camadas do MobileNetV2 (exceto BatchNormalization, que permanece congelada para estabilizar estatísticas). Para otimizar o desempenho, realizou-se grid search em learning rates {1e-4, 1e-5} × batch sizes {8, 32}. A melhor configuração (LR=1e-4, BS=8, 12 epochs totais: 4 head + 8 fine-tuning) alcançou desempenho em teste de Acurácia=0.848, F1=0.809 e ROC-AUC=0.931, demonstrando a eficácia do transfer learning com MobileNetV2 mesmo em cenários com dados limitados.

&emsp;&emsp;Essa abordagem é especialmente apropriada para cenários com dados geoespaciais limitados, reduzindo o risco de overfitting ao mesmo tempo que aumenta a capacidade discriminativa frente aos padrões espectrais-espaciais associados a mineralizações de terras raras.

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

**BAHRAMI, H.** et al. Machine Learning-Based Lithological Mapping from ASTER Remote-Sensing Imagery. *Minerals*, v. 14, n. 2, art. 202, 2024. DOI: 10.3390/min14020202. Disponível em: [https://doi.org/10.3390/min14020202](https://doi.org/10.3390/min14020202). Acesso em: 24 fev. 2026.

**CHEN, Y.; LIN, Z.; ZHAO, X.; WANG, G.; GU, Y.** Deep Learning-Based Classification of Hyperspectral Data. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, v. 7, n. 6, p. 2094–2107, 2014. DOI: 10.1109/JSTARS.2014.2329330. Disponível em: [https://doi.org/10.1109/JSTARS.2014.2329330](https://doi.org/10.1109/JSTARS.2014.2329330). Acesso em: 12 mar. 2026.

**LUO, Z.** et al. An explainable semi-supervised deep learning framework for mineral prospectivity mapping: DEEP-SEAM v1.0. *EGUsphere* (preprint), 2025. DOI: 10.5194/egusphere-2025-3283. Disponível em: [https://doi.org/10.5194/egusphere-2025-3283](https://doi.org/10.5194/egusphere-2025-3283). Acesso em: 23 fev. 2026.

**NATIONAL AERONAUTICS AND SPACE ADMINISTRATION (NASA).** ASTER L2 Surface Reflectance VNIR and Crosstalk-Corrected SWIR (AST_07XT) — Product Description. *NASA Earthdata*, s.d. Disponível em: [https://earthdata.nasa.gov/](https://earthdata.nasa.gov/). Acesso em: 26 fev. 2026.

**RAMSEY, M. S.; FLYNN, I. T. W.** The Spatial and Spectral Resolution of ASTER Infrared Image Data: A Paradigm Shift in Volcanological Remote Sensing. *Remote Sensing*, v. 12, n. 4, art. 738, 2020. DOI: 10.3390/rs12040738. Disponível em: [https://doi.org/10.3390/rs12040738](https://doi.org/10.3390/rs12040738). Acesso em: 26 fev. 2026.

**ROWAN, L. C.; MARS, J. C.** Lithologic mapping in the Mountain Pass, California area using ASTER data. *Remote Sensing of Environment*, v. 84, n. 3, p. 350–366, 2003. DOI: 10.1016/S0034-4257(02)00127-X. Disponível em: [https://doi.org/10.1016/S0034-4257(02)00127-X](https://doi.org/10.1016/S0034-4257(02)00127-X). Acesso em: 26 fev. 2026.

**SHIRMARD, H.; FARAHBAKHSH, E.; MÜLLER, R. D.; CHANDRA, R.** A review of machine learning in processing remote sensing data for mineral exploration. *Remote Sensing of Environment*, v. 268, art. 112750, 2022. DOI: 10.1016/j.rse.2021.112750. Disponível em: [https://doi.org/10.1016/j.rse.2021.112750](https://doi.org/10.1016/j.rse.2021.112750). Acesso em: 12 mar. 2026.

**SONG, Y.** et al. Predicting rare earth elements concentration in coal ashes with multi-task neural networks. *Materials Horizons*, v. 11, p. 747–757, 2024. DOI: 10.1039/D3MH01491F. Disponível em: [https://doi.org/10.1039/D3MH01491F](https://doi.org/10.1039/D3MH01491F). Acesso em: 23 fev. 2026.

**SUN, K.** et al. A Review of Mineral Prospectivity Mapping Using Deep Learning. *Minerals*, v. 14, n. 10, art. 1021, 2024. DOI: 10.3390/min14101021. Disponível em: [https://doi.org/10.3390/min14101021](https://doi.org/10.3390/min14101021). Acesso em: 26 fev. 2026.

**UNITED STATES GEOLOGICAL SURVEY (USGS).** Interior Department releases final 2025 List of Critical Minerals. *U.S. Geological Survey*, 14 nov. 2025. Disponível em: [https://www.usgs.gov/news/science-snippet/interior-department-releases-final-2025-list-critical-minerals](https://www.usgs.gov/news/science-snippet/interior-department-releases-final-2025-list-critical-minerals). Acesso em: 26 fev. 2026.