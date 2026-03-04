# Relatório de Ablation Study — CNN SpectraAI
---

## 1. Introdução

Um **ablation study** (estudo de ablação) é uma técnica do método científico aplicada ao desenvolvimento de modelos de Deep Learning. O princípio é simples: isolar e variar **um fator de cada vez**, mantendo todos os demais constantes, de forma a medir o impacto real de cada escolha arquitetural ou de hiperparâmetro.

Sem esse protocolo, é impossível saber se uma melhoria de acurácia veio do aumento de dropout, da mudança de learning rate, ou de ambos simultaneamente. A experimentação controlada garante **interpretação causal**, não apenas correlacional.

Este relatório documenta o design experimental, as configurações testadas, a análise de cada variável e os critérios para interpretação dos resultados.

---

## 2. Arquitetura Base da CNN

A CNN do projeto foi implementada em `src/models/cnn_builder.py` (branch `95-a5-cnn-arch-conv2d-pooling-dense-layers`, Larissa Souza) e posteriormente estendida com regularização em `101-regularization-code` (Larissa Souza, sobre base da Drielly Farias).

### 2.1 Estrutura das camadas

```
Input (128 × 128 × 9)
    │
    ├── Conv2D(32 filtros, kernel 3×3, padding='same', ReLU)
    │       + L2 regularization (kernel_regularizer)
    ├── Dropout(conv_dropout_rate)          ← adicionado na branch 101
    ├── MaxPooling2D(2×2, stride=2)
    │
    ├── Conv2D(64 filtros, kernel 3×3, padding='same', ReLU)
    │       + L2 regularization (kernel_regularizer)
    ├── Dropout(conv_dropout_rate)          ← adicionado na branch 101
    ├── MaxPooling2D(2×2, stride=2)
    │
    ├── Flatten
    ├── Dense(128 unidades, ReLU)
    ├── Dropout(dense_dropout_rate)
    │
    └── Output: Dense(2, sigmoid)           ← classificação binária
```

### 2.2 Entrada: 128×128×9

A entrada `128×128×9` reflete os **9 canais espectrais do sensor ASTER** (VNIR + SWIR) selecionados como relevantes para identificação de minerais associados a Terras Raras. Cada "pixel" do chip ASTER é uma janela espacial de 128×128 com informação multiespectral em 9 bandas.

---

## 3. Infraestrutura de Experimentação

O sistema utiliza uma arquitetura de configuração dinâmica via arquivos YAML, garantindo que o código do modelo permaneça isolado das variações de hiperparâmetros e facilitando a reprodutibilidade científica.

### 3.1 Pipeline de execução

O fluxo automatizado (implementado em src/models/) orquestra o experimento em três etapas:

Configuração: Definição centralizada de parâmetros (camadas, filtros, dropout, LR) em arquivos .yaml.

Processamento: A classe ExperimentRunner valida os dados, constrói a arquitetura via cnn_builder.py e executa o treinamento.

Persistência: Registro automático de pesos (.h5), histórico de métricas (history.json) e metadados do experimento em outputs/trained_models/.

### 3.2 Vantagens da Abordagem

Rastreabilidade: Cada rodada de treinamento está vinculada a um arquivo de configuração versionado no Git.

Isolamento de Variáveis: Permite alterar a agressividade da rede (ex: aumentar dropout) sem modificar a lógica base da CNN.

Escalabilidade: Estrutura preparada para execução de múltiplos experimentos em lote (batch processing).


### 3.3 Por que YAML?

O uso de arquivos YAML para configurar experimentos não é apenas uma conveniência — é um requisito de **reprodutibilidade científica**, princípio central do projeto (README, seção "Reprodutibilidade"). Cada arquivo YAML:

- É versionado no git junto com o código → o experimento pode ser replicado em qualquer máquina
- É salvo junto com os resultados em `config_used.json` → rastreabilidade total
- Isola variáveis → alterar um YAML não afeta o código do modelo
- Permite rodar N configurações sequencialmente sem intervenção manual

---

## 4. Configurações Comparadas

Foram definidas duas configurações para o ablation study. A variável de controle é clara: **apenas dropout e learning rate variam entre as configs**.

### 4.1 Tabela comparativa

| Hiperparâmetro | `baseline` | `higher_dropout` | Variação |
|---|---|---|---|
| `input_shape` | `[128, 128, 9]` | `[128, 128, 9]` | — |
| `num_classes` | `2` | `2` | — |
| `filters` | `[32, 64]` | `[32, 64]` | — |
| `kernel_size` | `3` | `3` | — |
| `pool_size` | `2` | `2` | — |
| `l2_regularizer` | `0.001` | `0.001` | — |
| `conv_dropout_rate` | **`0.2`** | **`0.3`** | +50% |
| `dense_dropout_rate` | **`0.5`** | **`0.6`** | +20% |
| `dense_units` | `128` | `128` | — |
| `batch_size` | `32` | `32` | — |
| `epochs` | `50` | `50` | — |
| `learning_rate` | **`0.001`** | **`0.0005`** | −50% |
| `optimizer` | `adam` | `adam` | — |

**Variáveis alteradas:** `conv_dropout_rate`, `dense_dropout_rate`, `learning_rate`
**Variáveis controladas:** todas as demais

---

## 5. Análise das Escolhas Arquiteturais

### 5.1 Regularização L2 nas convoluções

Adicionada na branch `101-regularization-code` como `kernel_regularizer=l2(0.001)` em ambas as camadas Conv2D.

**Mecanismo:** A penalidade L2 adiciona ao valor de loss o termo `λ · Σwᵢ²`, onde `λ = 0.001`. Isso força os pesos a permanecerem pequenos, reduzindo a capacidade do modelo de memorizar padrões específicos do conjunto de treino.

**Hipótese:** Com L2, espera-se redução do gap entre acurácia de treino e validação, às custas de uma acurácia de treino ligeiramente menor.

**Justificativa para este projeto:** Imagens ASTER de uma região geográfica específica têm alta correlação espacial. Sem regularização, a CNN pode aprender ruído geoespacial em vez de padrões mineralógicos genuínos.

### 5.2 Dropout nas camadas convolucionais

`conv_dropout_rate` é aplicado após cada bloco Conv2D+MaxPooling.

**Mecanismo:** Durante o treino, neurônios são desativados aleatoriamente com probabilidade `p`. Isso força redundância nas representações aprendidas e impede co-adaptação entre filtros.

**Diferença entre configs:**
- `baseline (0.2)`: 20% dos feature maps desativados por iteração → regularização leve
- `higher_dropout (0.3)`: 30% desativados → regularização mais agressiva nas convoluções

**Impacto esperado:** Dropout maior nas convoluções reduz a capacidade de memorização das camadas de extração de features. É especialmente útil quando o dataset tem tamanho limitado, que é o caso do projeto (imagens ASTER de região específica).

### 5.3 Dropout na camada densa

`dense_dropout_rate` é aplicado após a camada `Dense(128)`.

**Mecanismo:** Mesmo princípio, aplicado às ativações da camada fully-connected antes da saída.

**Diferença entre configs:**
- `baseline (0.5)`: dropout padrão, amplamente adotado na literatura (Srivastava et al., 2014)
- `higher_dropout (0.6)`: regularização mais intensa na etapa de classificação

**Risco:** Dropout muito alto na Dense pode prejudicar convergência se o modelo não tiver capacidade suficiente para aprender com 40% dos neurônios ativos.

### 5.4 Learning Rate

**Diferença entre configs:**
- `baseline (0.001)`: learning rate padrão do Adam, tipicamente o ponto de partida recomendado
- `higher_dropout (0.0005)`: learning rate reduzido à metade

**Justificativa da combinação:** Com mais dropout, o gradiente efetivo por passo é mais ruidoso. Um learning rate menor compensa esse ruído, permitindo que o otimizador dê passos menores e mais seguros. Esta é uma escolha de design coerente — não é uma variável isolada, mas uma consequência esperada do aumento de dropout.

> **Observação metodológica:** Alterar learning rate e dropout simultaneamente significa que `higher_dropout` testa um **conjunto de mudanças**, não uma variável isolada. Para um ablation puro, seria necessário uma terceira config com apenas o dropout aumentado e o LR mantido em 0.001, e uma quarta com apenas o LR reduzido. Recomenda-se adicionar essas configs em iterações futuras.

---

## 6. Métricas de Avaliação

Para comparação justa entre os experimentos, as seguintes métricas devem ser registradas ao final de cada run:

| Métrica | Fonte | Interpretação no contexto |
|---|---|---|
| `val_accuracy` final | `history.json` | Acurácia no conjunto de validação (20% dos dados) |
| `val_loss` final | `history.json` | Convergência — valores altos indicam overfitting ou underfitting |
| Gap `train_acc - val_acc` | `history.json` | Indicador de overfitting — ideal < 5% |
| Época de melhor `val_acc` | `history.json` | Velocidade de convergência |
| Estabilidade da curva `val_loss` | curva completa | Oscilação alta indica LR ou dropout inadequado |


## 7. Critérios de Interpretação

### 7.1 Se `higher_dropout` > `baseline` em `val_acc`

O modelo original estava sofrendo overfitting. A regularização mais agressiva e o LR menor permitiram melhor generalização. **Próximo passo:** testar `conv_dropout_rate=0.4` ou reduzir ainda mais o LR.

### 7.2 Se `baseline` > `higher_dropout` em `val_acc`

O modelo com dropout mais alto perdeu capacidade de aprendizado (underfitting). O dataset provavelmente não é grande o suficiente para suportar tanta regularização. **Próximo passo:** testar `conv_dropout_rate=0.25` (entre os dois).

### 7.3 Se os resultados forem equivalentes (`val_acc` com diferença < 1%)

Ambas as configurações têm poder de regularização suficiente. Nesse caso, preferir `baseline` por ter LR maior (converge mais rápido) e menos risco de underfitting.

### 7.4 Análise do gap treino/validação

| Gap | Diagnóstico |
|---|---|
| < 3% | Bem regularizado |
| 3–8% | Leve overfitting, aceitável |
| 8–15% | Overfitting moderado — aumentar regularização |
| > 15% | Overfitting severo — revisar arquitetura |

---

## 8. Configurações Futuras Sugeridas

Para expandir o ablation study de forma controlada, as seguintes configs são recomendadas:

```yaml
# config: deeper_network.yaml
# Hipótese: mais profundidade melhora extração de features espectrais
model:
  filters: [32, 64, 128]   # terceiro bloco convolucional
  ...

# config: smaller_input.yaml
# Hipótese: chips 64x64 são suficientes e reduzem custo computacional
model:
  input_shape: [64, 64, 9]
  ...

# config: higher_lr_only.yaml
# Hipótese: isolar o efeito do LR mantendo dropout do baseline
training:
  learning_rate: 0.0005
  # demais configs = baseline
  ...

# config: higher_dropout_only.yaml
# Hipótese: isolar o efeito do dropout mantendo LR do baseline
model:
  conv_dropout_rate: 0.3
  dense_dropout_rate: 0.6
  # demais configs = baseline
  ...
```

---

