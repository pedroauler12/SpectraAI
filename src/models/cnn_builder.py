"""
CNN Builder - Construção de Arquitetura de Rede Neural Convolucional.

Este módulo implementa a construção de modelos CNN (Convolutional Neural Networks)
com as camadas: Input -> Conv2D -> MaxPooling -> Flatten -> Dense.

Hiperparâmetros Documentados:
============================

1. CAMADA CONV2D - Primeira Convolução
   - Filtros: 32
   - Kernel Size: (3, 3)
   - Padding: 'same' (mantém dimensionalidade espacial)
   - Ativação: ReLU (acelera convergência e evita saturação de gradiente)
   - Justificativa: 32 filtros capturam features básicas; kernel 3x3 é padrão
     para extrair padrões locais eficientemente

2. CAMADA MAXPOOLING - Redução Dimensional (Primeira)
   - Pool Size: (2, 2)
   - Stride: 2 (deslocamento de 2 pixels)
   - Justificativa: Reduz número de parâmetros, captura features mais robustas,
     e melhora eficiência computacional

3. CAMADA CONV2D - Segunda Convolução
   - Filtros: 64
   - Kernel Size: (3, 3)
   - Padding: 'same'
   - Ativação: ReLU
   - Justificativa: 64 filtros capturam features mais complexas após primeira
     abstração; aproveita representação reduzida do pooling anterior

4. CAMADA MAXPOOLING - Redução Dimensional (Segunda)
   - Pool Size: (2, 2)
   - Stride: 2
   - Justificativa: Reduz ainda mais dimensionalidade antes do Flatten

5. CAMADA FLATTEN
   - Converte matriz multidimensional em vetor 1D
   - Justificativa: Necessária para conectar camadas convolucionais com Dense

6. CAMADA DENSE - Classificação
   - Unidades: Configurável (padrão 128)
   - Ativação: ReLU
   - Justificativa: Camada oculta para aprender representações não-lineares

7. CAMADA DE SAÍDA - Dense Final
   - Unidades: Número de classes (configurável)
   - Ativação: 'softmax' (multiclasse) ou 'sigmoid' (binária)
   - Justificativa: Produz distribuição de probabilidade sobre as classes

Referências:
- LeCun et al. (1998): Gradient-based learning applied to document recognition
- Krizhevsky et al. (2012): ImageNet Classification with Deep Convolutional Networks
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import Sequential, layers, regularizers


def build_cnn_model(
    input_shape: Tuple[int, int, int],
    n_classes: int,
    conv1_filters: int = 32,
    conv2_filters: int = 64,
    kernel_size: Tuple[int, int] = (3, 3),
    dense_units: int = 128,
    dropout_rate: float = 0.5,
    l2_regularizer: float = 0.001,
    conv_dropout_rate: float = 0.2,
    augmentation_layer: Optional[tf.keras.layers.Layer] = None,
) -> Sequential:
    """
    Constrói um modelo CNN com arquitetura:
    Input -> Conv2D (32) -> MaxPooling -> Conv2D (64) -> MaxPooling -> Flatten -> Dense -> Output

    Parâmetros
    ----------
    input_shape : Tuple[int, int, int]
        Forma da entrada (altura, largura, canais).
        Exemplo: (64, 64, 3) para imagens RGB 64x64

    n_classes : int
        Número de classes para classificação (>= 2)

    conv1_filters : int, default=32
        Número de filtros na primeira camada convolucional.
        Valores típicos: 16, 32, 64

    conv2_filters : int, default=64
        Número de filtros na segunda camada convolucional.
        Valores típicos: 32, 64, 128
        Geralmente maior que conv1_filters para capturar features mais complexas

    kernel_size : Tuple[int, int], default=(3, 3)
        Tamanho do kernel (filter) nas camadas Conv2D.
        3x3 é o tamanho padrão e recomendado

    dense_units : int, default=128
        Número de unidades na camada Dense oculta.
        Valores típicos: 64, 128, 256

    dropout_rate : float, default=0.5
        Taxa de dropout para regularização na camada Dense (evita overfitting).
        Valores típicos: 0.3 a 0.5
        Se 0, nenhum dropout é aplicado

    l2_regularizer : float, default=0.001
        Coeficiente de regularização L2 (penalidade de peso) para Conv2D e Dense.
        Valores típicos: 0.0001 a 0.01
        Se 0, nenhuma regularização L2 é aplicada

    conv_dropout_rate : float, default=0.2
        Taxa de dropout para camadas convolucionais (reduz overfitting em Conv2D).
        Valores típicos: 0.1 a 0.3 (menor que dropout_rate)
        Se 0, nenhum dropout é aplicado após Conv2D

    Retorna
    -------
    Sequential
        Modelo Keras compilado com otimizador Adam e loss adequado

    Exemplos
    --------
    Para classificação binária com imagens 64x64 RGB:

    >>> model = build_cnn_model(
    ...     input_shape=(64, 64, 3),
    ...     n_classes=2
    ... )
    >>> model.summary()

    Para classificação multiclasse (10 classes) com imagens 32x32 RGB:

    >>> model = build_cnn_model(
    ...     input_shape=(32, 32, 3),
    ...     n_classes=10,
    ...     conv1_filters=32,
    ...     conv2_filters=64
    ... )

    Para dados de imagem em escala de cinza (1 canal):

    >>> model = build_cnn_model(
    ...     input_shape=(64, 64, 1),
    ...     n_classes=3,
    ...     dense_units=256
    ... )
    """
    if n_classes < 2:
        raise ValueError("n_classes deve ser >= 2")

    if not (0 <= dropout_rate < 1):
        raise ValueError("dropout_rate deve estar entre 0 e 1")

    if not (0 <= l2_regularizer <= 1):
        raise ValueError("l2_regularizer deve estar entre 0 e 1")

    if not (0 <= conv_dropout_rate < 1):
        raise ValueError("conv_dropout_rate deve estar entre 0 e 1")

    layers_list = [
        # ========== INPUT LAYER ==========
        layers.Input(shape=input_shape),
    ]

    # ========== AUGMENTATION (opcional) ==========
    if augmentation_layer is not None:
        layers_list.append(augmentation_layer)

    layers_list.append(
        # ========== PRIMEIRA CONVOLUÇÃO COM REGULARIZAÇÃO L2 ==========
        layers.Conv2D(
            filters=conv1_filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_regularizer) if l2_regularizer > 0 else None,
            name='conv2d_1'
        ),
    )

    # Adicionar Dropout após primeira Conv se configurado
    if conv_dropout_rate > 0:
        layers_list.append(
            layers.Dropout(rate=conv_dropout_rate, name='dropout_conv1')
        )

    layers_list.extend([
        # ========== PRIMEIRO POOLING ==========
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='maxpooling2d_1'),

        # ========== SEGUNDA CONVOLUÇÃO COM REGULARIZAÇÃO L2 ==========
        layers.Conv2D(
            filters=conv2_filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_regularizer) if l2_regularizer > 0 else None,
            name='conv2d_2'
        ),
    ])

    # Adicionar Dropout após segunda Conv se configurado
    if conv_dropout_rate > 0:
        layers_list.append(
            layers.Dropout(rate=conv_dropout_rate, name='dropout_conv2')
        )

    layers_list.extend([
        # ========== SEGUNDO POOLING ==========
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='maxpooling2d_2'),

        # ========== FLATTEN ==========
        layers.Flatten(name='flatten'),

        # ========== CAMADA DENSA OCULTA COM REGULARIZAÇÃO L2 ==========
        layers.Dense(
            units=dense_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_regularizer) if l2_regularizer > 0 else None,
            name='dense_hidden'
        ),

        # ========== DROPOUT NA CAMADA DENSA ==========
        layers.Dropout(rate=dropout_rate, name='dropout_dense'),

        # ========== CAMADA DE SAÍDA COM REGULARIZAÇÃO L2 ==========
        layers.Dense(
            units=n_classes,
            activation='softmax' if n_classes > 2 else 'sigmoid',
            kernel_regularizer=regularizers.l2(l2_regularizer) if l2_regularizer > 0 else None,
            name='output'
        ),
    ])

    model = Sequential(layers_list)

    # ========== COMPILAÇÃO DO MODELO ==========
    # Otimizador: Adam (adapta taxa de aprendizado por parâmetro)
    # Loss: categoricalcrossentropy (multiclasse) ou binária
    # Métrica: acurácia
    loss = 'categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy'

    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )

    return model


def build_deep_cnn_model(
    input_shape: Tuple[int, int, int],
    n_classes: int,
    filters_list: list = None,
    kernel_size: Tuple[int, int] = (3, 3),
    dense_units: int = 128,
    dropout_rate: float = 0.5,
    l2_regularizer: float = 0.001,
    conv_dropout_rate: float = 0.2,
) -> Sequential:
    """
    Constrói um modelo CNN com profundidade variável (N blocos convolucionais).

    Cada bloco convolucional consiste em:
        Conv2D -> Dropout (opcional) -> MaxPooling2D

    Parâmetros
    ----------
    input_shape : Tuple[int, int, int]
        Forma da entrada (altura, largura, canais).
    n_classes : int
        Número de classes para classificação (>= 2).
    filters_list : list
        Lista com número de filtros por bloco convolucional.
        Ex: [32, 64, 128] cria 3 blocos Conv2D.
    kernel_size : Tuple[int, int], default=(3, 3)
        Tamanho do kernel nas camadas Conv2D.
    dense_units : int, default=128
        Unidades na camada Dense oculta.
    dropout_rate : float, default=0.5
        Taxa de dropout na camada Dense.
    l2_regularizer : float, default=0.001
        Coeficiente de regularização L2.
    conv_dropout_rate : float, default=0.2
        Taxa de dropout após cada camada Conv2D.

    Retorna
    -------
    Sequential
        Modelo Keras compilado.
    """
    if filters_list is None:
        filters_list = [32, 64, 128]

    if n_classes < 2:
        raise ValueError("n_classes deve ser >= 2")

    reg = regularizers.l2(l2_regularizer) if l2_regularizer > 0 else None

    layers_list = [layers.Input(shape=input_shape)]

    for i, n_filters in enumerate(filters_list, start=1):
        layers_list.append(
            layers.Conv2D(
                filters=n_filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu',
                kernel_regularizer=reg,
                name=f'conv2d_{i}',
            )
        )
        if conv_dropout_rate > 0:
            layers_list.append(
                layers.Dropout(rate=conv_dropout_rate, name=f'dropout_conv{i}')
            )
        layers_list.append(
            layers.MaxPooling2D(pool_size=(2, 2), strides=2, name=f'maxpooling2d_{i}')
        )

    layers_list.extend([
        layers.Flatten(name='flatten'),
        layers.Dense(
            units=dense_units,
            activation='relu',
            kernel_regularizer=reg,
            name='dense_hidden',
        ),
        layers.Dropout(rate=dropout_rate, name='dropout_dense'),
        layers.Dense(
            units=n_classes,
            activation='softmax' if n_classes > 2 else 'sigmoid',
            kernel_regularizer=reg,
            name='output',
        ),
    ])

    model = Sequential(layers_list)

    loss = 'categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    return model


def load_backbone(
    input_shape_3ch: Tuple[int, int, int],
) -> Tuple[tf.keras.Model, bool]:
    """
    Carrega MobileNetV2 pré-treinado no ImageNet como backbone para transfer learning.

    A camada top (classificação) é removida e o backbone é **totalmente congelado**.
    Para descongelar camadas para fine-tuning, use ``unfreeze_backbone_layers``.

    Parâmetros
    ----------
    input_shape_3ch : Tuple[int, int, int]
        Shape esperado pelo backbone (H, W, 3).

    Retorna
    -------
    Tuple[tf.keras.Model, bool]
        (backbone, pretrained) — o modelo e se os pesos ImageNet foram carregados.
    """
    try:
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape_3ch,
        )
        pretrained = True
    except Exception as exc:
        print(f"Falha ao carregar pesos pretrained ({exc}). Usando pesos aleatorios.")
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=input_shape_3ch,
        )
        pretrained = False

    backbone.trainable = False
    return backbone, pretrained


def unfreeze_backbone_layers(
    model: tf.keras.Model,
    fine_tune_last_layers: int = 20,
    learning_rate: float = 1e-5,
) -> dict:
    """
    Desbloqueia as últimas camadas do backbone para fine-tuning e recompila o modelo.

    Implementa a fase 2 do pipeline de transfer learning: após treinar o head
    com backbone congelado, esta função desbloqueia as últimas N camadas do
    backbone (exceto BatchNormalization) e recompila com learning rate menor.

    Parâmetros
    ----------
    model : tf.keras.Model
        Modelo construído por ``build_transfer_model``.
    fine_tune_last_layers : int, default=20
        Número de camadas finais do backbone a descongelar.
    learning_rate : float, default=1e-5
        Learning rate para a fase de fine-tuning (tipicamente 10x menor que a fase 1).

    Retorna
    -------
    dict
        Resumo com total de camadas, camadas descongeladas e parâmetros treináveis.
    """
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise ValueError("Backbone MobileNetV2 não encontrado no modelo.")

    backbone.trainable = True
    total_layers = len(backbone.layers)
    unfrozen_names = []

    for layer in backbone.layers[:-fine_tune_last_layers]:
        layer.trainable = False

    for layer in backbone.layers[-fine_tune_last_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen_names.append(layer.name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )

    trainable_count = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_count = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )

    return {
        "backbone_total_layers": total_layers,
        "fine_tune_last_layers": fine_tune_last_layers,
        "unfrozen_layer_names": unfrozen_names,
        "n_unfrozen": len(unfrozen_names),
        "learning_rate": float(learning_rate),
        "trainable_params": int(trainable_count),
        "non_trainable_params": int(non_trainable_count),
    }


def build_transfer_model(
    input_shape: Tuple[int, int, int],
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.25,
) -> Tuple[tf.keras.Model, dict]:
    """
    Constrói modelo de transfer learning com MobileNetV2 para classificação binária.

    O backbone inicia **totalmente congelado** (fase 1 — head training).
    Para a fase 2 (fine-tuning), use ``unfreeze_backbone_layers``.

    Arquitetura:
        Input (H, W, 9) → Conv2D 1×1 (9→3) → BN → ReLU → MobileNetV2 → GAP → Dropout → Dense(1, sigmoid)

    Parâmetros
    ----------
    input_shape : Tuple[int, int, int]
        Shape de entrada (H, W, C). Ex: (160, 160, 9).
    learning_rate : float, default=1e-4
        Taxa de aprendizado do otimizador Adam para a fase 1.
    dropout_rate : float, default=0.25
        Taxa de dropout antes da camada de predição.

    Retorna
    -------
    Tuple[tf.keras.Model, dict]
        (model, info) — modelo compilado e dicionário com metadados da arquitetura.
    """
    backbone, pretrained = load_backbone(
        (input_shape[0], input_shape[1], 3),
    )

    inputs = tf.keras.Input(shape=input_shape, name="aster_9ch_input")
    x = layers.Conv2D(3, kernel_size=1, padding="same", use_bias=False, name="channel_adapter")(inputs)
    x = layers.BatchNormalization(name="channel_adapter_bn")(x)
    x = layers.Activation("relu", name="channel_adapter_relu")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = tf.keras.Model(inputs, outputs, name="a08_transfer_learning_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )

    info = {
        "backbone": "MobileNetV2",
        "pretrained_loaded": pretrained,
        "phase": "head_training",
        "input_shape": tuple(int(v) for v in input_shape),
        "learning_rate": float(learning_rate),
        "dropout_rate": float(dropout_rate),
    }
    return model, info


def get_model_architecture_summary(model: Sequential) -> dict:
    """
    Retorna informações resumidas sobre a arquitetura do modelo CNN.

    Parâmetros
    ----------
    model : Sequential
        Modelo Keras já construído

    Retorna
    -------
    dict
        Dicionário com informações: total_params, trainable_params, 
        non_trainable_params, layers_info
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    layers_info = []
    for layer in model.layers:
        # Tentar obter output_shape de forma robusta
        try:
            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape
            elif hasattr(layer, 'output'):
                output_shape = tuple(layer.output.shape)
            else:
                output_shape = 'Unknown'
        except:
            output_shape = 'Unknown'
        
        layers_info.append({
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': output_shape,
            'params': layer.count_params()
        })

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'layers_info': layers_info
    }


if __name__ == '__main__':
    # Exemplo de uso
    print("Construindo modelo CNN...")

    # Exemplo 1: Classificação binária com imagens 64x64 RGB
    model = build_cnn_model(
        input_shape=(64, 64, 3),
        n_classes=2
    )
    print("\n=== Modelo CNN (Binária) ===")
    model.summary()

    # Exemplo 2: Classificação multiclasse (5 classes)
    model_multi = build_cnn_model(
        input_shape=(64, 64, 3),
        n_classes=5,
        dense_units=256
    )
    print("\n=== Modelo CNN (Multiclasse) ===")
    model_multi.summary()

    # Resumo da arquitetura
    print("\n=== Resumo de Hiperparâmetros ===")
    print(get_model_architecture_summary(model_multi))
