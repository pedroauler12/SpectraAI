# SpectraAI: Prospecção de Terras Raras a partir de Imagens Multiespectrais ASTER com Aprendizado de Máquina e Visão Computacional

### Autores: Drielly Santana Farias, Eduardo Farias Rizk, Giovanna Fátima de Britto Vieira, Larissa Martins Pereira de Souza, Lucas Ramenzoni Jorge,  Mateus Beppler Pereira, Pedro Auler de Barros Martins

## 1. Introdução (citações corrigidas)

&emsp;&emsp; Os Elementos Terras Raras (Rare Earth Elements — REE) compõem um grupo de 17 elementos amplamente empregados em tecnologias de alto valor agregado, incluindo eletrônica, aplicações industriais avançadas e sistemas energéticos. A relevância econômica e estratégica desses elementos tem sido reiterada por órgãos oficiais e relatórios setoriais recentes, que destacam vulnerabilidades em cadeias globais de suprimento e riscos associados a alta concentração geográfica de produção e refino (UNITED STATES GEOLOGICAL SURVEY, 2025).

&emsp;&emsp; Do ponto de vista operacional, a prospecção mineral tradicional depende de campanhas de campo, amostragem e análises laboratoriais, etapas onerosas e de difícil escalabilidade espacial. Em contrapartida, o sensoriamento remoto oferece um meio de observação sistemática e repetível para apoiar a triagem de alvos, especialmente quando combinado a métodos quantitativos de análise de dados. Em particular, a exploração mineral por sensoriamento remoto se beneficia da relação entre resposta espectral e mineralogia/alteração, permitindo inferências indiretas sobre litologias e processos geológicos associados a mineralizações.

&emsp;&emsp; Nesse contexto, o Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) consolidou-se como um dos sensores mais utilizados em mapeamento litológico e exploração mineral por disponibilizar bandas espectrais relevantes em VNIR e SWIR, além de histórico robusto de aplicações documentadas na literatura (ABRAMS; YAMAGUCHI, 2019; RAMSEY; FLYNN, 2020). No entanto, a interpretação manual de cenas multiespectrais permanece limitada pela alta dimensionalidade espectral, heterogeneidade espacial e pela sutileza de padrões associados a mineralizações, o que pode introduzir subjetividade e restringir a reprodutibilidade dos resultados.

&emsp;&emsp; Diante disso, este trabalho apresenta uma proposta metodológica inicial para construção de um pipeline de ciência de dados geoespaciais, utilizando imagens ASTER e dados de referência fornecidos pela Frontera Minerals, com o objetivo de transformar as cenas em um conjunto supervisionado de amostras rotuladas e avaliar modelos de aprendizado de máquina e visão computacional para estimar, de forma probabilística, o potencial prospectivo em áreas de interesse. A proposta privilegia a reprodutibilidade do processamento e a geração de evidências quantitativas que possam apoiar, em ciclos posteriores, validação geológica e refinamento do método.

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

A base de dados do projeto é composta por amostras de solo e rocha coletadas in situ pela **Frontera Minerals**, contendo teores geoquímicos de Elementos de Terras Raras (ETR). As amostras foram rotuladas binariamente:

* **Classe Positiva (y = 1):** Áreas com teores acima do *cut-off* econômico, associadas a depósitos iônicos ou rochas alcalinas mineralizadas.
* **Classe Negativa (y = 0):** Áreas estéreis ou com teores de base (background).

As assinaturas espectrais foram extraídas de imagens do sensor **ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer)**, utilizando as bandas do visível e infravermelho (VNIR) e infravermelho de ondas curtas (SWIR), com resolução espacial reamostrada para compatibilidade.

#### 3.2.2. Pré-processamento e Engenharia de Atributos

Para mitigar ruídos e isolar a resposta mineralógica, o pipeline de dados executou:

1. **Filtragem de Máscaras:** Remoção de pixels contaminados por nuvens e densa cobertura vegetal (NDVI > limiar).
2. **Cálculo de Índices Minerais:** Foram geradas *features* baseadas em razões de bandas consagradas na literatura de sensoriamento remoto mineral, como o **Índice de Argilas** $[B06 / (B05 + B04)]$, visando destacar produtos de alteração hidrotermal e intemperismo.
3. **Vetorização:** Cada amostra foi convertida em um vetor de alta dimensionalidade ($p = 147.456$), representando tanto as bandas brutas quanto as janelas espaciais adjacentes.


#### 3.2.3 Geração de amostras (chips) e rotulagem supervisionada

&emsp;&emsp; O dataset supervisionado é construído a partir de chips gerados ao redor de pontos georreferenciados. Cada chip é um GeoTIFF multibanda com bandas VNIR+SWIR empilhadas e alinhadas espacialmente. O recorte usa bbox com jitter controlado por semente, garantindo que o ponto de referência permaneça dentro do chip.

&emsp;&emsp; Em seguida, os chips são convertidos para um dataset tabular, no qual cada amostra é representada por um vetor de pixels (`pixel_*`) e metadados (path, dimensões, CRS etc.). A rotulagem é aplicada por mapeamento de `image_id` para listas de positivos e negativos fornecidas no `extracted_codes.json`.

#### 3.2.3. Protocolo de Divisão de Dados (Anti-Leakage)

Um ponto crítico da metodologia é o controle de **vazamento de dados (spatial leakage)**. A divisão do dataset em treino (60%), validação (20%) e teste (20%) foi realizada no nível de cena (**image_id**).

* Amostras pertencentes à mesma imagem de satélite foram mantidas obrigatoriamente no mesmo grupo.
* Utilizou-se o método **StratifiedGroupKFold** para garantir que a proporção de classes fosse mantida em todos os *folds*, impedindo que o modelo memorizasse condições de iluminação ou sensores específicos de uma única imagem.


#### 3.2.4. Modelagem Clássica (Baseline)

Foram avaliados três algoritmos de aprendizado supervisionado para estabelecer o desempenho de referência:

1. **Support Vector Machine (SVM):** Implementada com kernel linear e regularização , visando a maximização da margem em espaço de alta dimensionalidade.
2. **Random Forest (RF):** Conjunto de árvores de decisão para capturar interações não lineares entre as bandas espectrais.
3. **Regressão Logística:** Baseline linear para verificação de separabilidade simples e calibração probabilística.

A otimização de hiperparâmetros foi realizada via **GridSearchCV**, utilizando o conjunto de validação para a escolha final dos modelos.

#### 3.2.5. Protocolo de Avaliação e Calibração de Limiar

Dada a natureza exploratória do problema, o limiar de decisão ($\tau$) não foi fixado em 0.5. Em vez disso:

1. O modelo gerou scores contínuos.
2. O limiar ótimo foi selecionado no conjunto de validação através da maximização do **F1-Score** na curva Precision-Recall.
3. As métricas finais (F1, Precision, Recall, ROC-AUC e PR-AUC) foram calculadas exclusivamente no conjunto de teste isolado.

$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### 3.2.6 Critérios de avaliação e produto final esperado

&emsp;&emsp; A avaliação considera métricas de classificação binária: acurácia, precisão, revocação (recall), F1-score, balanced accuracy, ROC-AUC e PR-AUC, além de matriz de confusão e análise de erros. O resultado pode ser interpretado como escore prospectivo por amostra/região, permitindo ordenar áreas por probabilidade estimada de classe positiva.

&emsp;&emsp; O produto esperado nesta etapa é um mecanismo reprodutível de geração de chips, montagem de dataset e inferência supervisionada, servindo como base para refinamentos metodológicos e expansão para modelos de visão computacional nas próximas sprints.

## 4. Trabalhos Relacionados

#### Trabalho Relacionado 1: Twenty Years of ASTER Contributions to Lithologic Mapping and Mineral Exploration

&emsp;&emsp; O artigo de revisão "Twenty Years of ASTER Contributions to Lithologic Mapping and Mineral Exploration", publicado por Abrams e Yamaguchi (2019), resume o histórico de aplicações bem-sucedidas do sensor ASTER na pesquisa e mapeamento mineral. Lançado em 1999, o ASTER revolucionou a exploração geológica global ao fornecer melhor resolução espacial e capacidades multiespectrais únicas, apresentando seis bandas no infravermelho de ondas curtas (SWIR) e cinco bandas no infravermelho termal (TIR).

&emsp;&emsp; Essa configuração espectral superou as limitações de satélites anteriores, como o Landsat, permitindo a distinção precisa de grupos minerais diagnósticos de alteração hidrotermal — como argilas, carbonatos, sulfatos e distinções na composição de silicatos. No contexto geológico voltado para minerais críticos e de Terras Raras, a revisão de Abrams e Yamaguchi destaca trabalhos pioneiros, como o estudo de Rowan e Mars (2003), que foram os primeiros a demonstrar a capacidade das 14 bandas do ASTER em distinguir litologias e mapear zonas de contato metamórfico associadas a depósitos de minerais de terras raras na região de Mountain Pass, Califórnia.

&emsp;&emsp; A revisão literária também aborda a evolução das técnicas aplicadas ao extenso volume de imagens do ASTER para extração de informações mineralógicas:os autores relatam o uso bem-sucedido de técnicas mais simples, como índices minerais baseados em razões de bandas (band ratios), até métodos de processamento estatístico, como Análise de Componentes Principais (PCA). Ademais, o artigo relata o uso crescente de métodos analíticos sofisticados nos últimos anos, incluindo machine learning e modelos de redes neurais (como as redes neurais MLP e modelos SOM) utilizados para classificar complexidades espaciais e realizar mapeamentos litológicos e de zonas de alteração.

&emsp;&emsp; Essa trajetória documentada por Abrams e Yamaguchi (2019) corrobora o problema e a justificativa metodológica que escolhemos. O artigo confirma que as imagens ASTER possuem dados  suficientes para caracterizar as assinaturas espectrais associadas a depósitos minerais. No entanto, a alta dimensionalidade e a complexidade espacial desses dados tornam a análise manual desafiadora, especialmente para padrões sutis. Dessa forma, o histórico literário valida a criação do pipeline de ciência de dados e o uso de algoritmos de Deep Learning e Visão Computacional, atestando a viabilidade técnica de utilizar os dados multiespectrais ASTER como a principal fonte de evidências para estimar e rankear áreas prospectivas de forma mais objetiva, escalável e probabilística.

#### Trabalho Relacionado: Machine Learning-Based Lithological Mapping from ASTER Remote-Sensing Imagery

&emsp;&emsp; Um avanço recente e relevante é o estudo de Bahrami et al. (2024), que investiga mapeamento litológico automatizado a partir de imagens ASTER por meio de uma comparação sistemática entre algoritmos de machine learning tradicionais (Random Forest, SVM, Gradient Boosting e XGBoost) e uma abordagem de deep learning (ANN) aplicada ao caso da região mineralizada de Sar-Cheshmeh (Irã). O trabalho se destaca por estruturar um pipeline comparável ao de exploração mineral baseada em sensoriamento remoto, incorporando engenharia/seleção de atributos espectrais (features derivadas de bandas e análise de correlação/importance) e avaliando quantitativamente o desempenho dos modelos via acurácia global para diferentes classes litológicas. ([MDPI][1])
&emsp;&emsp; Como contribuição para este projeto, Bahrami et al. reforçam que o ASTER mantém alta utilidade para tarefas de classificação litológica e identificação indireta de minerais quando combinado com métodos supervisionados, além de evidenciar que escolhas de pré-processamento e seleção de variáveis afetam significativamente a qualidade do mapa final. ([MDPI][1])
&emsp;&emsp; Entretanto, há limitações importantes quando comparamos com a proposta da Frontera Minerals. Primeiro, o estudo é orientado a classes litológicas em um contexto regional específico, não sendo desenhado diretamente para um problema de “detecção/ranking prospectivo” (ex.: presença/ausência de assinatura associada a Terras Raras em torno de ocorrências conhecidas). Segundo, o trabalho depende de um conjunto de treinamento bem definido para classes do mapeamento local, enquanto o desafio do projeto envolve generalização e rotulagem positiva/negativa por proximidade geográfica (chips ao redor de coordenadas de referência), o que tende a introduzir ruído de rótulo e exigir estratégias de validação e modelagem. Ainda assim, o artigo oferece um baseline metodológico sólido para justificar a etapa de comparação entre modelos clássicos e redes neurais usando ASTER, além de servir de referência para decisões de features e avaliação.

### Trabalho Relacionado: Redes Neurais para Prospecção de Terras Raras

&emsp;&emsp;Avançando além da caracterização espectral dos sensores, a integração de modelos baseados em aprendizado profundo (_Deep Learning_) surge como o passo evolutivo necessário para superar a sutileza das assinaturas de elementos de terras raras (REE). O trabalho de Luo et al. (2025) introduz o framework **DEEP-SEAM v1.0**, demonstrando que a natureza não linear e altamente heterogênea dos conjuntos de dados de exploração impõe limitações aos métodos tradicionais de mapeamento.

&emsp;&emsp;Para solucionar essa complexidade, os autores empregam redes neurais para extrair padrões ocultos em dados multifonte, aplicando a Deviation Network (DevNet) para identificar anomalias mesmo em cenários de dados esparsos e desbalanceados. Complementando essa visão técnica, o estudo consolidado de Song et al. (2023) reforça que redes multitarefa podem filtrar ruídos e descobrir correlações não lineares entre composições químicas e a presença de REEs, reduzindo gargalos de custo e tempo em relação às análises estatísticas convencionais.

&emsp;&emsp;Essa convergência entre modelos _data-driven_ e a necessidade de interpretar assinaturas minerais complexas corrobora a adoção de redes neurais no SpectraAI. Ao utilizar redes neurais e visão computacional para processar imagens ASTER, o projeto promove o ranqueamento de áreas prospectivas de terras raras de forma escalável, objetiva e com alta fidelidade geológica.

### Trabalho Relacionado: Classificação de Uso e Cobertura do Solo via CNNs e Visão Computacional

&emsp;&emsp;A transição do processamento puramente espectral para a análise contextual-espacial é amplamente validada no estado da arte da classificação de uso e cobertura do solo (LULC - Land Use and Land Cover). O estudo de Tong et al. (2020), intitulado "Land-cover classification with high-resolution remote sensing images using transferable deep models", estabeleceu um marco ao demonstrar que Redes Neurais Convencionais (CNNs) superam algoritmos clássicos (como SVM e Random Forest) ao capturar não apenas a reflectância individual do pixel, mas a dependência espacial entre vizinhos. Utilizando arquiteturas profundas, os autores provaram que a extração automática de feições espaciais reduz drasticamente o ruído de classificação (efeito "sal e pimenta", onde aparecem pixels com valores alternadamente modificados para 0 (pimenta) e para o valor máximo da imagem (sal)) comum em métodos tradicionais de sensoriamento remoto.

&emsp;&emsp;Ao comparar o trabalho de Tong et al. com a proposta do SpectraAI, identificam-se pontos de convergência e distinções fundamentais:

&emsp;&emsp;Enquanto o estudo de SOTA foca em feições macroscópicas e antropogênicas (padrões urbanos, texturas de plantações e corpos d'água), o SpectraAI busca padrões mineralógicos sutis e sub-pixel. No entanto, a lógica de "análise por vizinhança" (patches/chips) validada por Tong é o que justifica a evolução do nosso pipeline de modelos MLP/SVM para CNNs.

&emsp;&emsp;A superioridade das CNNs na LULC reside na capacidade de aprender "texturas". Para a prospecção de Terras Raras, isso sugere que o modelo não deve olhar apenas para a cor do pixel, mas para o gradiente espectral da zona de alteração hidrotermal em que o mineral se formou e hoje está inserido.

&emsp;&emsp;Dessa forma, a literatura de uso do solo fornece a base teórica para o uso de chips multiespectrais no SpectraAI: se a visão computacional consegue distinguir texturas complexas de solo e vegetação, ela possui o potencial necessário para identificar as assinaturas espaciais discretas de depósitos de ETR que modelos baseados apenas em tabelas de pixels tendem a ignorar.


## 5. Proposta Metodológica Preliminar

  Como proposta preliminar, o projeto estrutura a transformação das cenas ASTER em amostras padronizadas (“chips” multiespectrais) rotuladas em classes positivas e negativas a partir do *ground truth* fornecido. Em seguida, avalia-se um conjunto inicial de modelos supervisionados, abrangendo baselines clássicos e alternativas baseadas em redes neurais, com foco em generalização e redução de subjetividade na interpretação. A saída esperada é um escore ou probabilidade por amostra/região, permitindo o ranqueamento de áreas prospectivas para posterior validação geológica e refinamento do método nas próximas Sprints.


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

**TONG, X.** et al. Land-cover classification with high-resolution remote sensing images using transferable deep models. *Remote Sensing of Environment*, v. 237, 111322, 2020. DOI: 10.1016/j.rse.2019.111322. Disponível em: [https://doi.org/10.1016/j.rse.2019.111322](https://doi.org/10.1016/j.rse.2019.111322). Acesso em: 06 mar. 2026.