# SpectraAI: Prospecção de Terras Raras a partir de Imagens Multiespectrais ASTER com Aprendizado de Máquina e Visão Computacional

### Autores: Drielly Santana Farias, Eduardo Farias Rizk, Giovanna Fátima de Britto Vieira, Larissa Martins Pereira de Souza, Lucas Ramenzoni Jorge,  Mateus Beppler Pereira, Pedro Auler de Barros Martins

## 1. Introdução

&emsp;&emsp; Os Elementos Terras Raras (Rare Earth Elements — REE) compõem um grupo de 17 elementos amplamente empregados em tecnologias de alto valor agregado, incluindo eletrônica, aplicações industriais avançadas e sistemas energéticos. A relevância econômica e estratégica desses elementos tem sido reiterada por órgãos oficiais e relatórios setoriais recentes, que destacam vulnerabilidades em cadeias globais de suprimento e riscos associados a alta concentração geográfica de produção e refino (UNITED STATES GEOLOGICAL SURVEY, 2025).

&emsp;&emsp; Do ponto de vista operacional, a prospecção mineral tradicional depende de campanhas de campo, amostragem e análises laboratoriais, etapas onerosas e de difícil escalabilidade espacial. Em contrapartida, o sensoriamento remoto oferece um meio de observação sistemática e repetível para apoiar a triagem de alvos, especialmente quando combinado a métodos quantitativos de análise de dados. Em particular, a exploração mineral por sensoriamento remoto se beneficia da relação entre resposta espectral e mineralogia/alteração, permitindo inferências indiretas sobre litologias e processos geológicos associados a mineralizações.

&emsp;&emsp; Nesse contexto, o Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) consolidou-se como um dos sensores mais utilizados em mapeamento litológico e exploração mineral por disponibilizar bandas espectrais relevantes em VNIR e SWIR, além de histórico robusto de aplicações documentadas na literatura (ABRAMS; YAMAGUCHI, 2019; RAMSEY; FLYNN, 2020). No entanto, a interpretação manual de cenas multiespectrais permanece limitada pela alta dimensionalidade espectral, heterogeneidade espacial e pela sutileza de padrões associados a mineralizações, o que pode introduzir subjetividade e restringir a reprodutibilidade dos resultados.

&emsp;&emsp; Diante disso, este trabalho apresenta uma proposta metodológica inicial para construção de um pipeline de ciência de dados geoespaciais, utilizando imagens ASTER e dados de referência fornecidos pela Frontera Minerals, com o objetivo de transformar as cenas em um conjunto supervisionado de amostras rotuladas e avaliar modelos de aprendizado de máquina e visão computacional para estimar, de forma probabilística, o potencial prospectivo em áreas de interesse. A proposta privilegia a reprodutibilidade do processamento e a geração de evidências quantitativas que possam apoiar, em ciclos posteriores, validação geológica e refinamento do método.

&emsp;&emsp;Historicamente, grande parte das aplicações de aprendizado de máquina em sensoriamento remoto mineral tem sido conduzida a partir de representações tabulares dos dados espectrais, nas quais cada pixel ou amostra é descrito como um vetor de atributos derivados das bandas disponíveis. Nesse contexto, modelos clássicos e redes neurais rasas, como Multi-Layer Perceptrons (MLP), foram amplamente empregados para tarefas de classificação litológica ou identificação de assinaturas minerais, sobretudo por sua capacidade de modelar relações não lineares entre variáveis espectrais. Entretanto, essa abordagem apresenta uma limitação importante: ao tratar cada amostra como um vetor independente, a estrutura espacial presente nas imagens multiespectrais é, em grande parte, descartada. Em problemas geológicos, essa informação espacial pode ser relevante, uma vez que processos de alteração mineral, zonas de contato litológico e padrões geomorfológicos tendem a se manifestar como estruturas contínuas ou texturas distribuídas no espaço.

&emsp;&emsp;Diante dessa limitação, abordagens baseadas em visão computacional têm sido cada vez mais exploradas em dados de sensoriamento remoto. Em particular, redes neurais convolucionais (Convolutional Neural Networks — CNN) permitem aprender automaticamente padrões espaciais e espectrais a partir de janelas de imagem, preservando relações de vizinhança entre pixels e capturando estruturas que dificilmente seriam representadas por atributos tabulares isolados. No contexto de prospecção mineral, essa capacidade pode contribuir para identificar padrões sutis associados a zonas de alteração ou assinaturas espectrais distribuídas espacialmente, potencialmente ampliando a capacidade de generalização dos modelos.

&emsp;&emsp;Assim, além da avaliação inicial de modelos supervisionados clássicos como baseline, este trabalho também considera a perspectiva de evolução metodológica para abordagens baseadas em visão computacional. A hipótese central é que a incorporação de informação espacial por meio de representações em forma de chips multiespectrais possa permitir a extração automática de características relevantes, fornecendo uma base mais adequada para modelagem preditiva em tarefas de mapeamento prospectivo de elementos de terras raras.

#### 1.1 Objetivos da Pesquisa
##### Objetivo Geral

&emsp;&emsp;Desenvolver e avaliar um modelo de Deep Learning aplicado à visão computacional capaz de analisar imagens multiespectrais do sensor ASTER e estimar, de forma probabilística, o potencial de ocorrência de elementos de Terras Raras (REE), produzindo um ranking de prospectividade mineral que auxilie a priorização de áreas para campanhas de pesquisa geológica.

##### Objetivos Específicos

A partir desse objetivo geral, delineiam-se os seguintes objetivos específicos:

Realizar engenharia de atributos espectrais, investigando transformações e combinações de bandas do ASTER capazes de destacar assinaturas espectrais relacionadas a minerais de alteração hidrotermal (argilas, óxidos e carbonatos), frequentemente associados a sistemas mineralizados.

Desenvolver e treinar modelos CNN, com foco em arquiteturas de visão computacional capazes de extrair padrões espaciais e espectrais presentes nas imagens multiespectrais.

Avaliar o desempenho dos modelos treinados utilizando áreas com ocorrências conhecidas de Terras Raras, por meio das métricas f1-score e AUC-ROC.

Produzir um ranking de áreas com maior potencial prospectivo, permitindo priorizar regiões para futuras etapas de pesquisa mineral e campanhas de campo.

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

&emsp;&emsp; O material de referência para supervisão é composto por coordenadas georreferenciadas de interesse geológico (incluindo Serra Verde e CBMM) e por listas de códigos positivos e negativos fornecidas pelo parceiro. Esses insumos orientam a associação entre amostras e rótulos no dataset final, constituindo o *ground truth* operacional utilizado nos experimentos desta sprint.

### 3.2 Métodos

#### 3.2.1. Aquisição de Dados e Alvos Geológicos

A base de dados do projeto é composta por amostras de solo e rocha coletadas *in situ* pela **Frontera Minerals**, contendo teores geoquímicos de Elementos de Terras Raras (ETR). As amostras foram rotuladas binariamente:

* **Classe Positiva (y = 1):** Áreas com teores acima do *cut-off* econômico, associadas a depósitos iônicos ou rochas alcalinas mineralizadas.
* **Classe Negativa (y = 0):** Áreas estéreis ou com teores de base (*background*).

As assinaturas espectrais foram extraídas de imagens do sensor **ASTER**, utilizando as bandas do visível e infravermelho (VNIR) e infravermelho de ondas curtas (SWIR). Devido à degradação do sensor SWIR após 2008, o pipeline prioriza cenas históricas (2000-2007) com cobertura de nuvens inferior a 20% (conforme documentado no protocolo de acesso ASTER), garantindo a integridade dos dados para mapeamento de argilas.

#### 3.2.2. Pré-processamento e Engenharia de Atributos

Para mitigar ruídos e isolar a resposta mineralógica, o pipeline executou:

1. **Filtragem de Máscaras:** Remoção de pixels contaminados por nuvens e densa cobertura vegetal (NDVI > limiar).
2. **Reprojeção:** Conversão sistemática de coordenadas para WGS84, corrigindo discrepâncias entre os dados de campo (SAD69) e os produtos orbitais.
3. **Cálculo de Índices Minerais:** Geração de *features* baseadas em razões de bandas como o **Índice de Argilas**: 
$$\text{Índice de Argilas} = \frac{B06}{B05 + B04}$$
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

#### 3.2.6. Protocolo de Divisão de Dados e Anti-Leakage

O controle de **vazamento de dados (spatial leakage)** é rigoroso. A divisão em treino (60%), validação (20%) e teste (20%) é feita no nível de **cena (image_id)**. Amostras da mesma imagem permanecem no mesmo grupo, utilizando **StratifiedGroupKFold** para manter a proporção de classes em todos os folds e impedir que o modelo memorize condições específicas de uma única captura.

#### 3.2.7. Protocolo de Avaliação e Produto Final

O limiar de decisão ($\tau$) não é fixado em 0.5, sendo otimizado no conjunto de validação para maximizar o **F1-Score** na curva Precision-Recall. As métricas finais incluem Acurácia, Precisão, Recall, F1-score, Balanced Accuracy, ROC-AUC e PR-AUC. O produto final é um mecanismo reprodutível que permite ordenar áreas por probabilidade estimada, servindo como base para a expansão da prospecção mineral da **Frontera Minerals**.


#### 3.2.8 Arquitetura da CNN e Hiperparâmetros

&emsp;&emsp; Para a etapa de visão computacional foi implementada uma Rede Neural Convolucional (CNN) simples, projetada como arquitetura inicial de experimentação sobre os chips multiespectrais ASTER. A rede recebe como entrada tensores correspondentes aos chips extraídos das cenas, preservando a estrutura espacial e espectral das imagens.

&emsp;&emsp; A arquitetura segue uma configuração sequencial composta por blocos convolucionais e camadas densas. Cada bloco convolucional é formado por uma camada Conv2D, seguida de função de ativação ReLU e camada de max pooling, responsável pela redução progressiva da dimensionalidade espacial e pela extração de padrões locais relevantes. Após os blocos convolucionais, o tensor resultante é achatado (flatten) e processado por camadas densas responsáveis pela etapa de classificação binária (presença ou ausência de assinatura associada a áreas prospectivas).

&emsp;&emsp; A camada final utiliza função de ativação sigmoid, produzindo uma probabilidade associada à classe positiva. Esse valor é posteriormente utilizado no ranqueamento de áreas prospectivas.

&emsp;&emsp; Os principais hiperparâmetros utilizados no treinamento da CNN são apresentados na Tabela 1.

Tabela 1 – Hiperparâmetros utilizados no treinamento da CNN

| Hiperparâmetro     | Valor                                    |
| ------------------ | ---------------------------------------- |
| Learning Rate      | 0.001                                    |
| Batch Size         | 32                                       |
| Número de Epochs   | 50                                       |
| Otimizador         | Adam                                     |
| Função de perda    | Binary Cross-Entropy                     |
| Função de ativação | ReLU (camadas internas), Sigmoid (saída) |

Esses valores foram definidos inicialmente com base em práticas comuns em tarefas de classificação de imagens e serão refinados em experimentos futuros por meio de estratégias de ajuste de hiperparâmetros.

#### 3.2.9 Protocolo Experimental

&emsp;&emsp;O protocolo experimental foi estruturado para avaliar a capacidade dos modelos em identificar padrões associados à presença de elementos de terras raras a partir de dados multiespectrais ASTER. O conjunto de dados foi dividido em três subconjuntos: treinamento (60%), validação (20%) e teste (20%), respeitando o agrupamento por image_id para evitar vazamento de informação entre conjuntos.

&emsp;&emsp;Durante o treinamento da CNN, o conjunto de validação foi utilizado para monitorar o desempenho do modelo e auxiliar na seleção do limiar de decisão e de configurações de treinamento. Ao final do processo, o modelo com melhor desempenho no conjunto de validação foi aplicado ao conjunto de teste, que permaneceu isolado durante todo o processo de desenvolvimento.

&emsp;&emsp;A avaliação de desempenho considera métricas para capturar diferentes aspectos da qualidade da classificação, incluindo F1-score e ROC-AUC . Essas métricas permitem analisar tanto a capacidade geral de classificação quanto o comportamento do modelo em cenários com possível desbalanceamento entre classes.

&emsp;&emsp;Para garantir a reprodutibilidade computacional do pipeline, todas as etapas de processamento, geração de chips, engenharia de atributos e treinamento dos modelos foram implementadas em ambiente Python utilizando bibliotecas amplamente adotadas em ciência de dados geoespaciais e aprendizado de máquina.As etapas de modelagem supervisionada e avaliação utilizaram scikit-learn e TensorFlow/Keras. O pipeline foi estruturado de forma modular, permitindo a repetição sistemática dos experimentos a partir de parâmetros controlados, incluindo sementes aleatórias fixas para geração de amostras e divisão de dados. Essa abordagem garante consistência entre execuções e facilita a replicação dos resultados em estudos futuros ou em novas áreas de interesse geológico.
## 4. Trabalhos Relacionados

#### Trabalho Relacionado 1: Twenty Years of ASTER Contributions to Lithologic Mapping and Mineral Exploration

&emsp;&emsp; O artigo de revisão "Twenty Years of ASTER Contributions to Lithologic Mapping and Mineral Exploration", publicado por Abrams e Yamaguchi (2019), resume o histórico de aplicações bem-sucedidas do sensor ASTER na pesquisa e mapeamento mineral. Lançado em 1999, o ASTER revolucionou a exploração geológica global ao fornecer melhor resolução espacial e capacidades multiespectrais únicas, apresentando seis bandas no infravermelho de ondas curtas (SWIR) e cinco bandas no infravermelho termal (TIR).

&emsp;&emsp; Essa configuração espectral superou as limitações de satélites anteriores, como o Landsat, permitindo a distinção precisa de grupos minerais diagnósticos de alteração hidrotermal — como argilas, carbonatos, sulfatos e distinções na composição de silicatos. No contexto geológico voltado para minerais críticos e de Terras Raras, a revisão de Abrams e Yamaguchi destaca trabalhos pioneiros, como o estudo de Rowan e Mars (2003), que foram os primeiros a demonstrar a capacidade das 14 bandas do ASTER em distinguir litologias e mapear zonas de contato metamórfico associadas a depósitos de minerais de terras raras na região de Mountain Pass, Califórnia.

&emsp;&emsp; A revisão literária também aborda a evolução das técnicas aplicadas ao extenso volume de imagens do ASTER para extração de informações mineralógicas:os autores relatam o uso bem-sucedido de técnicas mais simples, como índices minerais baseados em razões de bandas (band ratios), até métodos de processamento estatístico, como Análise de Componentes Principais (PCA). Ademais, o artigo relata o uso crescente de métodos analíticos sofisticados nos últimos anos, incluindo machine learning e modelos de redes neurais (como as redes neurais MLP e modelos SOM) utilizados para classificar complexidades espaciais e realizar mapeamentos litológicos e de zonas de alteração.

&emsp;&emsp; Essa trajetória documentada por Abrams e Yamaguchi (2019) corrobora o problema e a justificativa metodológica que escolhemos. O artigo confirma que as imagens ASTER possuem dados  suficientes para caracterizar as assinaturas espectrais associadas a depósitos minerais. No entanto, a alta dimensionalidade e a complexidade espacial desses dados tornam a análise manual desafiadora, especialmente para padrões sutis. Dessa forma, o histórico literário valida a criação do pipeline de ciência de dados e o uso de algoritmos de Deep Learning e Visão Computacional, atestando a viabilidade técnica de utilizar os dados multiespectrais ASTER como a principal fonte de evidências para estimar e rankear áreas prospectivas de forma mais objetiva, escalável e probabilística.

&emsp;&emsp;**Análise Crítica:** Embora Abrams e Yamaguchi (2019) validem o potencial do ASTER, a revisão foca predominantemente em métodos de interpretação visual ou estatística clássica. O SpectraAI avança ao propor a automação dessa interpretação via Deep Learning, reduzindo a dependência da expertise subjetiva do analista humano mencionada pelos autores.

#### Trabalho Relacionado: Machine Learning-Based Lithological Mapping from ASTER Remote-Sensing Imagery

&emsp;&emsp; Um avanço recente e relevante é o estudo de Bahrami et al. (2024), que investiga mapeamento litológico automatizado a partir de imagens ASTER por meio de uma comparação sistemática entre algoritmos de machine learning tradicionais (Random Forest, SVM, Gradient Boosting e XGBoost) e uma abordagem de deep learning (ANN) aplicada ao caso da região mineralizada de Sar-Cheshmeh (Irã). O trabalho se destaca por estruturar um pipeline comparável ao de exploração mineral baseada em sensoriamento remoto, incorporando engenharia/seleção de atributos espectrais (features derivadas de bandas e análise de correlação/importance) e avaliando quantitativamente o desempenho dos modelos via acurácia global para diferentes classes litológicas. ([MDPI][1])

&emsp;&emsp; Como contribuição para este projeto, Bahrami et al. reforçam que o ASTER mantém alta utilidade para tarefas de classificação litológica e identificação indireta de minerais quando combinado com métodos supervisionados, além de evidenciar que escolhas de pré-processamento e seleção de variáveis afetam significativamente a qualidade do mapa final. ([MDPI][1])

&emsp;&emsp; Entretanto, há limitações importantes quando comparamos com a proposta da Frontera Minerals. Primeiro, o estudo é orientado a classes litológicas em um contexto regional específico, não sendo desenhado diretamente para um problema de “detecção/ranking prospectivo” (ex.: presença/ausência de assinatura associada a Terras Raras em torno de ocorrências conhecidas). Segundo, o trabalho depende de um conjunto de treinamento bem definido para classes do mapeamento local, enquanto o desafio do projeto envolve generalização e rotulagem positiva/negativa por proximidade geográfica (chips ao redor de coordenadas de referência), o que tende a introduzir ruído de rótulo e exigir estratégias de validação e modelagem. Ainda assim, o artigo oferece um baseline metodológico sólido para justificar a etapa de comparação entre modelos clássicos e redes neurais usando ASTER, além de servir de referência para decisões de features e avaliação.

&emsp;&emsp;**Análise Crítica:** A lacuna identificada no trabalho de Bahrami et al. (2024) reside na abordagem puramente tabular. Ao ignorar o contexto espacial adjacente ao pixel, o modelo perde a continuidade geológica. Nosso projeto resolve essa limitação através do uso de chips espaciais e convoluções, preservando a textura do terreno.

### Trabalho Relacionado: Redes Neurais para Prospecção de Terras Raras

&emsp;&emsp;Avançando além da caracterização espectral dos sensores, a integração de modelos baseados em aprendizado profundo (_Deep Learning_) surge como o passo evolutivo necessário para superar a sutileza das assinaturas de elementos de terras raras (REE). O trabalho de Luo et al. (2025) introduz o framework **DEEP-SEAM v1.0**, demonstrando que a natureza não linear e altamente heterogênea dos conjuntos de dados de exploração impõe limitações aos métodos tradicionais de mapeamento.

&emsp;&emsp;Para solucionar essa complexidade, os autores empregam redes neurais para extrair padrões ocultos em dados multifonte, aplicando a Deviation Network (DevNet) para identificar anomalias mesmo em cenários de dados esparsos e desbalanceados. Complementando essa visão técnica, o estudo consolidado de Song et al. (2023) reforça que redes multitarefa podem filtrar ruídos e descobrir correlações não lineares entre composições químicas e a presença de REEs, reduzindo gargalos de custo e tempo em relação às análises estatísticas convencionais.

&emsp;&emsp;Essa convergência entre modelos _data-driven_ e a necessidade de interpretar assinaturas minerais complexas corrobora a adoção de redes neurais no SpectraAI. Ao utilizar redes neurais e visão computacional para processar imagens ASTER, o projeto promove o ranqueamento de áreas prospectivas de terras raras de forma escalável, objetiva e com alta fidelidade geológica.

&emsp;&emsp;**Análise Crítica:** O framework DEEP-SEAM v1.0 (Luo et al., 2025) foca em dados multifonte complexos. O SpectraAI diferencia-se por buscar uma solução otimizada especificamente para o sensor ASTER em áreas de solo exposto, criando um especialista de domínio em imagens orbitais antes de escalar para a fusão de dados.

## 5. Proposta Metodológica Preliminar

  Como proposta preliminar, o projeto estrutura a transformação das cenas ASTER em amostras padronizadas (“chips” multiespectrais) rotuladas em classes positivas e negativas a partir do *ground truth* fornecido. Em seguida, avalia-se um conjunto inicial de modelos supervisionados, abrangendo baselines clássicos e alternativas baseadas em redes neurais, com foco em generalização e redução de subjetividade na interpretação. A saída esperada é um escore ou probabilidade por amostra/região, permitindo o ranqueamento de áreas prospectivas para posterior validação geológica e refinamento do método nas próximas Sprints.


## 6. Resultados Preliminares

### 6.1 Baseline Clássico (A02)

&emsp;&emsp; Na etapa de modelagem clássica, três algoritmos foram avaliados como baselines supervisionados: Random Forest, SVM (kernel linear) e Regressão Logística. Os modelos foram treinados sobre vetores de 9 médias espectrais por banda (VNIR+SWIR) e avaliados no conjunto de teste com threshold otimizado via maximização do F1-Score no conjunto de validação.

&emsp;&emsp; Entre os baselines, o SVM com kernel linear obteve o melhor F1-Score (0.851), seguido pela Regressão Logística (0.818) e pelo Random Forest (0.780). O SVM também apresentou o melhor recall (0.870), indicando maior capacidade de capturar depósitos reais. O Random Forest, por sua vez, obteve a maior ROC-AUC (0.930), sugerindo boa capacidade discriminativa geral, embora com menor recall no threshold otimizado.

### 6.2 MLP Baseline (A03)

&emsp;&emsp; O baseline neural consiste em uma rede MLP com duas camadas ocultas (32 e 16 neurônios, ativação ReLU) e camada de saída com 2 neurônios (sigmoid), treinada com sparse categorical crossentropy e otimizador Adam. A entrada são as mesmas 9 médias espectrais por banda utilizadas nos baselines clássicos, sem PCA.

&emsp;&emsp; O modelo foi treinado por até 100 épocas com Early Stopping (paciência de 10 épocas, monitorando val_loss), utilizando batch size de 32 e divisão treino/validação/teste de 60%/20%/20% estratificada por imagem. O threshold de decisão foi otimizado via F1 no conjunto de validação.

### 6.3 Comparação

&emsp;&emsp; A comparação quantitativa entre os modelos permite avaliar se a capacidade de modelar relações não-lineares da MLP oferece ganhos sobre os baselines clássicos no contexto de prospecção mineral. As métricas detalhadas e visualizações comparativas estão disponíveis no notebook do artefato A03 (`artefatos/a03_mlp_baseline/a03_mlp_baseline.ipynb`), incluindo gráficos de barras agrupadas e análise de trade-offs entre precisão e recall para cada modelo.

&emsp;&emsp; Os resultados indicam que, neste regime de poucos dados (177 amostras de treino) e representação simplificada (médias por banda), os modelos clássicos e a MLP operam em faixas de desempenho comparáveis, com diferenças que dependem da métrica priorizada. A análise completa das implicações operacionais é apresentada no notebook.

## 7. Discussão e Próximos Passos

&emsp;&emsp; Os resultados obtidos até o momento demonstram que tanto modelos clássicos quanto a MLP baseline conseguem discriminar, com desempenho acima do aleatório, áreas com e sem potencial prospectivo para ETR a partir de assinaturas espectrais ASTER. No entanto, a representação atual — médias por banda — descarta informação espacial e textural que pode ser diagnóstica para identificação de mineralizações, constituindo a principal limitação arquitetural desta etapa.

&emsp;&emsp; O regime de poucos dados (177 amostras de treino) e a ausência de validação geográfica cruzada impõem cautela na interpretação dos resultados. Os modelos podem estar capturando correlações espúrias associadas a condições de iluminação ou contexto geológico compartilhado entre treino e teste, ao invés de padrões espectrais genuinamente associados a mineralizações de ETR.

&emsp;&emsp; Para as próximas sprints, propõe-se: (i) migração para arquiteturas convolucionais (CNNs) que processem os chips 128×128×9 completos, preservando informação espacial; (ii) técnicas de data augmentation (rotação, flip, jitter espectral) para expandir o N efetivo; (iii) transfer learning a partir de datasets maiores de sensoriamento remoto; (iv) validação espacial cruzada para avaliar generalização geográfica; e (v) fusão com dados geológicos complementares para um ranqueamento prospectivo multifonte.

### Referências

**ABRAMS, M.; YAMAGUCHI, Y.** Twenty Years of ASTER Contributions to Lithologic Mapping and Mineral Exploration. *Remote Sensing*, v. 11, n. 11, 1394, 2019. DOI: 10.3390/rs11111394. Disponível em: [https://doi.org/10.3390/rs11111394](https://doi.org/10.3390/rs11111394). Acesso em: 22 fev. 2026.

**BAHRAMI, H.** et al. Machine Learning-Based Lithological Mapping from ASTER Remote-Sensing Imagery. *Minerals*, v. 14, n. 2, 202, 2024. DOI: 10.3390/min14020202. Disponível em: [https://doi.org/10.3390/min14020202](https://doi.org/10.3390/min14020202). Acesso em: 24 fev. 2026.

**LUO, Z.** et al. An explainable semi-supervised deep learning framework for mineral prospectivity mapping: DEEP-SEAM v1.0. *EGUsphere* (preprint), 2025. DOI: 10.5194/egusphere-2025-3283. Disponível em: [https://doi.org/10.5194/egusphere-2025-3283](https://doi.org/10.5194/egusphere-2025-3283). Acesso em: 23 fev. 2026.

**NATIONAL AERONAUTICS AND SPACE ADMINISTRATION (NASA).** ASTER L2 Surface Reflectance VNIR and Crosstalk-Corrected SWIR (AST_07XT) — Product Description. *NASA Earthdata*, s.d. Disponível em: [https://earthdata.nasa.gov/](https://earthdata.nasa.gov/). Acesso em: 26 fev. 2026.

**RAMSEY, M. S.; FLYNN, I. T. W.** The Spatial and Spectral Resolution of ASTER Infrared Image Data: A Paradigm Shift in Volcanological Remote Sensing. *Remote Sensing*, v. 12, n. 4, 738, 2020. DOI: 10.3390/rs12040738. Disponível em: [https://www.mdpi.com/2072-4292/12/4/738](https://www.mdpi.com/2072-4292/12/4/738). Acesso em: 26 fev. 2026.

**ROWAN, L. C.; MARS, J. C.** Lithologic mapping in the Mountain Pass, California area using ASTER data. *Remote Sensing of Environment*, v. 84, n. 3, p. 350–366, 2003. DOI: 10.1016/S0034-4257(02)00127-X. Disponível em: [https://doi.org/10.1016/S0034-4257(02)00127-X](https://doi.org/10.1016/S0034-4257%2802%2900127-X). Acesso em: 26 fev. 2026.

**SONG, Y.** et al. Predicting rare earth elements concentration in coal ashes with multi-task neural networks. *Materials Horizons*, 2024. DOI: 10.1039/D3MH01491F. Disponível em: [https://doi.org/10.1039/D3MH01491F](https://doi.org/10.1039/D3MH01491F). Acesso em: 23 fev. 2026.

**SUN, K.** et al. A Review of Mineral Prospectivity Mapping Using Deep Learning. *Minerals*, v. 14, n. 10, 1021, 2024. DOI: 10.3390/min14101021. Disponível em: [https://www.mdpi.com/2075-163X/14/10/1021](https://www.mdpi.com/2075-163X/14/10/1021). Acesso em: 26 fev. 2026.

**UNITED STATES GEOLOGICAL SURVEY (USGS).** Interior Department releases final 2025 List of Critical Minerals. *U.S. Geological Survey*, 14 nov. 2025. Disponível em: [https://www.usgs.gov/news/science-snippet/interior-department-releases-final-2025-list-critical-minerals](https://www.usgs.gov/news/science-snippet/interior-department-releases-final-2025-list-critical-minerals). Acesso em: 26 fev. 2026.