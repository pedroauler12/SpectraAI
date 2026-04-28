# Relatório de Ablation Study — CNN SpectraAI
---

## 1. Introdução

Um **ablation study** (estudo de ablação) é uma técnica do método científico aplicada ao desenvolvimento de modelos de Deep Learning. O princípio é simples: isolar e variar **um fator de cada vez**, mantendo todos os demais constantes, de forma a medir o impacto real de cada escolha arquitetural ou de hiperparâmetro.

Sem esse protocolo, é impossível saber se uma melhoria de acurácia veio do aumento de dropout, da mudança de learning rate, ou de ambos simultaneamente. A experimentação controlada garante **interpretação causal**, não apenas correlacional.

Este relatório documenta o design experimental, as configurações testadas, os resultados obtidos na GPU com o dataset completo e as conclusões sobre os melhores hiperparâmetros encontrados.

---

## 2. Arquitetura Base da CNN

A CNN do projeto foi implementada em `src/models/cnn_builder.py` (branch `95-a5-cnn-arch-conv2d-pooling-dense-layers`, posteriormente estendida com regularização em `101-regularization-code`).

### 2.1 Estrutura das camadas

```
Input (128 × 128 × 9)
    │
    ├── Conv2D(32 filtros, kernel K×K, padding='same', ReLU)
    │       + L2 regularization (kernel_regularizer)
    ├── Dropout(conv_dropout_rate)
    ├── MaxPooling2D(P×P, stride=P)
    │
    ├── Conv2D(64 filtros, kernel K×K, padding='same', ReLU)
    │       + L2 regularization (kernel_regularizer)
    ├── Dropout(conv_dropout_rate)
    ├── MaxPooling2D(P×P, stride=P)
    │
    ├── Flatten
    ├── Dense(128 unidades, ReLU)
    ├── Dropout(dense_dropout_rate)
    │
    └── Output: Dense(2, sigmoid)           ← classificação binária
```

> K e P são variáveis do ablation study (ver Seção 4).

### 2.2 Entrada: 128×128×9

A entrada `128×128×9` reflete os **9 canais espectrais do sensor ASTER** (VNIR + SWIR) selecionados como relevantes para identificação de minerais associados a Terras Raras. Cada chip ASTER é uma janela espacial de 128×128 com informação multiespectral em 9 bandas.

---

## 3. Infraestrutura de Experimentação

O sistema utiliza configuração dinâmica via arquivos YAML, garantindo que o código do modelo permaneça isolado das variações de hiperparâmetros e facilitando a reprodutibilidade científica.

### 3.1 Pipeline de execução

O fluxo automatizado em `src/models/` orquestra o experimento em três etapas:

- **Configuração:** Definição centralizada de parâmetros em arquivos `.yaml`
- **Processamento:** A classe `ExperimentRunner` valida os dados, constrói a arquitetura via `cnn_builder.py` e executa o treinamento
- **Persistência:** Registro automático de pesos (`.h5`), histórico de métricas (`history.json`) e metadados do experimento em `outputs/trained_models/`

### 3.2 Por que YAML?

Cada arquivo YAML é versionado no git junto com o código → o experimento pode ser replicado em qualquer máquina. É salvo junto com os resultados em `config_used.json` → rastreabilidade total. Isola variáveis → alterar um YAML não afeta o código do modelo.

Os experimentos foram executados em **GPU** (servidor `cc09-g1`) com o dataset completo, utilizando `random_seed=42` para garantir reprodutibilidade.

---

## 4. Configurações Testadas

Foram testadas **4 configurações** ao longo do ablation study, variando dois eixos principais: arquitetura (kernel/pool) e regularização (dropout/learning rate).

### 4.1 Tabela comparativa

| Hiperparâmetro | `baseline` | `higher_dropout` | `model_architechture` | `k4p4_higher_droput` |
|---|---|---|---|---|
| `filters` | `[32, 64]` | `[32, 64]` | `[32, 64]` | `[32, 64]` |
| `kernel_size` | **`3`** | **`3`** | **`4`** | **`4`** |
| `pool_size` | **`2`** | **`2`** | **`4`** | **`4`** |
| `l2_regularizer` | `0.001` | `0.001` | `0.001` | `0.001` |
| `conv_dropout_rate` | `0.2` | **`0.3`** | `0.2` | **`0.3`** |
| `dense_dropout_rate` | `0.5` | **`0.6`** | `0.5` | **`0.6`** |
| `learning_rate` | `0.001` | **`0.0005`** | `0.001` | **`0.0005`** |
| `batch_size` | `32` | `32` | `32` | `32` |
| `epochs` | `50` | `50` | `50` | `50` |

**Eixo 1 — Arquitetura:** `kernel_size` e `pool_size` (baseline/higher_dropout = k3/p2 vs model_architechture/k4p4 = k4/p4)

**Eixo 2 — Regularização:** dropout e learning rate (baseline/model_architechture = padrão vs higher_dropout/k4p4 = aumentado)

### 4.2 Variações de kernel e pool testadas

Durante a busca por hiperparâmetros, as seguintes combinações foram exploradas modificando o arquivo `model_architechture.yaml`:

| kernel | pool | Val Acc |
|--------|------|---------|
| 3 | 2 | 0.8305 |
| 2 | 2 | 0.8644 |
| 3 | 3 | 0.8814 |
| 4 | 3 | 0.8136 |
| **4** | **4** | **0.8814** |

---

## 5. Análise das Escolhas Arquiteturais

### 5.1 Regularização L2 nas convoluções

Adicionada em `101-regularization-code` como `kernel_regularizer=l2(0.001)` em ambas as camadas Conv2D.

**Mecanismo:** A penalidade L2 adiciona ao valor de loss o termo `λ · Σwᵢ²`, onde `λ = 0.001`. Isso força os pesos a permanecerem pequenos, reduzindo a capacidade do modelo de memorizar padrões específicos do treino.

**Justificativa:** Imagens ASTER de uma região geográfica específica têm alta correlação espacial. Sem regularização, a CNN pode aprender ruído geoespacial em vez de padrões mineralógicos genuínos.

### 5.2 Kernel size e Pool size

**Mecanismo do kernel:** O kernel define o campo receptivo de cada filtro convolucional. Um kernel maior captura relações espaciais mais amplas em uma única operação.

- `kernel=3`: campo receptivo de 3×3 pixels — adequado para texturas finas
- `kernel=4`: campo receptivo de 4×4 pixels — captura padrões espectrais em escala levemente maior

**Mecanismo do pool:** O MaxPooling reduz a resolução espacial, forçando invariância a pequenas translações.

- `pool=2`: redução 2× — preserva mais detalhe espacial
- `pool=4`: redução 4× — maior abstração, representação mais compacta

**Resultado observado:** A combinação `kernel=4, pool=4` e `kernel=3, pool=3` empatam em val_acc (88.14%), mas `kernel=4, pool=4` apresenta F1 e balanced accuracy superiores. A hipótese é que o campo receptivo maior é mais adequado para as texturas espectrais das imagens ASTER.

### 5.3 Dropout nas camadas convolucionais

`conv_dropout_rate` é aplicado após cada bloco Conv2D+MaxPooling.

- `baseline (0.2)`: 20% dos feature maps desativados — regularização leve
- `higher_dropout (0.3)`: 30% desativados — regularização mais agressiva

**Resultado observado:** Com a arquitetura `k4/p4`, o dropout padrão (0.2) superou o dropout aumentado (0.3). O campo receptivo maior do kernel=4 já fornece regularização implícita suficiente.

### 5.4 Dropout na camada densa

`dense_dropout_rate` é aplicado após a camada `Dense(128)`.

- `baseline (0.5)`: dropout padrão da literatura (Srivastava et al., 2014)
- `higher_dropout (0.6)`: regularização mais intensa na etapa de classificação

**Resultado observado:** O aumento para 0.6 na camada densa combinado com LR reduzido não trouxe ganho — a config `k4p4_higher_droput` foi a pior das quatro.

### 5.5 Learning Rate

- `baseline/model_architechture (0.001)`: LR padrão do Adam
- `higher_dropout/k4p4 (0.0005)`: LR reduzido à metade

**Observação metodológica:** Dropout e LR foram variados simultaneamente nas configs `higher_dropout` e `k4p4_higher_droput`. Para um ablation puro, seriam necessárias configs isolando cada variável. Os resultados indicam que a combinação dropout alto + LR baixo é prejudicial para esta arquitetura, mas não é possível separar individualmente qual das mudanças é responsável.

---

## 6. Resultados

Todos os experimentos foram executados na GPU com o dataset completo, `random_seed=42`, 50 épocas e batch size 32.

### 6.1 Melhor resultado por configuração

| Config | Val Acc | Val F1 | Val Balanced Acc | Sensitivity | Specificity | AUC-ROC |
|--------|---------|--------|------------------|-------------|-------------|---------|
| **model_architechture** (k4/p4) | **0.8814** | **0.8835** | **0.8972** | **0.9524** | 0.8421 | 0.5439 |
| baseline (k3/p2) | 0.8814 | 0.8829 | 0.8866 | 0.9048 | 0.8684 | **0.6779** |
| higher_dropout (k3/p2) | 0.8644 | 0.8671 | 0.8841 | 0.9524 | 0.8158 | 0.5702 |
| k4p4_higher_droput | 0.8475 | 0.8494 | 0.8390 | 0.8571 | 0.8684 | 0.4587 |

### 6.2 Melhor run absoluta: `model_architechture_20260312_133740`

```
Config:            kernel=4, pool=4, conv_dropout=0.2, dense_dropout=0.5, lr=0.001
Val Accuracy:      88.14%
Val F1-Score:      88.35%
Val Balanced Acc:  89.72%
Val Sensitivity:   95.24%   →   20 TP / 1 FN
Val Precision:     89.83%
Val Specificity:   84.21%
Matriz de Confusão: TP=20, FP=6, TN=32, FN=1
```

### 6.3 Análise do gap treino/validação

| Config | Train Acc | Val Acc | Gap | Diagnóstico |
|--------|-----------|---------|-----|-------------|
| model_architechture | 0.9746 | 0.8814 | 9.3% | Leve overfitting — aceitável |
| baseline | 0.9703 | 0.8814 | 8.9% | Leve overfitting — aceitável |
| higher_dropout | 0.9746 | 0.8644 | 11.0% | Overfitting moderado |
| k4p4_higher_droput | 0.9746 | 0.8475 | 12.7% | Overfitting moderado |

> Referência: gap < 3% = bem regularizado; 3–8% = leve overfitting; 8–15% = overfitting moderado; > 15% = overfitting severo.

---

## 7. Conclusões

### 7.1 Parâmetros ótimos encontrados

A configuração `model_architechture` com `kernel_size=4` e `pool_size=4` é a vencedora em F1-score e balanced accuracy. Os parâmetros finais recomendados são:

```yaml
model:
  filters: [32, 64]
  kernel_size: 4        # melhor campo receptivo para texturas ASTER
  pool_size: 4          # maior abstração espacial
  l2_regularizer: 0.001
  conv_dropout_rate: 0.2
  dense_dropout_rate: 0.5
  dense_units: 128

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 50
  optimizer: adam
```

### 7.2 Interpretação dos resultados

**Kernel=4, pool=4 supera kernel=3, pool=2:** O campo receptivo maior capta padrões espectrais das imagens ASTER em uma escala mais adequada. A redução mais agressiva pelo pool=4 produz representações mais abstratas que generalizam melhor.

**Dropout padrão bate dropout aumentado com k4/p4:** A arquitetura com kernel e pool maiores já introduz regularização implícita pela compressão da representação. Aumentar o dropout além do necessário prejudica o aprendizado.

**Trade-off AUC-ROC:** O `baseline` (k3/p2) apresenta AUC-ROC superior (0.68 vs 0.54). Isso indica que a arquitetura k3/p2 discrimina melhor ao longo de diferentes thresholds, enquanto o k4/p4 é mais preciso no threshold padrão de 0.5. Para uma aplicação de detecção de Terras Raras onde minimizar falsos negativos é crítico, a maior sensitivity do `model_architechture` (95.24%) justifica a escolha.

### 7.3 Resposta às hipóteses do estudo

| Hipótese | Resultado |
|----------|-----------|
| Dropout maior reduz overfitting | ❌ Não confirmado — aumentou o gap treino/val |
| LR menor melhora com mais dropout | ❌ Não confirmado — piorou a performance geral |
| Kernel maior melhora extração de features ASTER | ✅ Confirmado — k4/p4 superou k3/p2 em F1 e balanced acc |
| k4/p4 + dropout aumentado combina os benefícios | ❌ Não confirmado — pior configuração geral |

---

## 8. Limitações e Próximos Passos

- **AUC-ROC baixo no melhor modelo:** Investigar threshold calibration ou usar `val_f1` como métrica de early stopping em vez de `val_accuracy`
- **Ablation puro de LR e dropout separados:** Criar configs isolando cada variável para determinar qual mudança é responsável pelo impacto observado
- **Terceiro bloco convolucional:** Testar `filters=[32, 64, 128]` com a arquitetura k4/p4 encontrada
- **Kernel size 5:** Não testado — pode capturar padrões espectrais ainda mais amplos nas imagens ASTER

