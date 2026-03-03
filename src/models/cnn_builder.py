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
from tensorflow.keras import Sequential, layers


def build_cnn_model(
    input_shape: Tuple[int, int, int],
    n_classes: int,
    conv1_filters: int = 32,
    conv2_filters: int = 64,
    kernel_size: Tuple[int, int] = (3, 3),
    dense_units: int = 128,
    dropout_rate: float = 0.5,
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
        Taxa de dropout para regularização (evita overfitting).
        Valores típicos: 0.3 a 0.5
        Se 0, nenhum dropout é aplicado

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

    model = Sequential([
        # ========== INPUT LAYER ==========
        # Define a forma dos dados de entrada
        layers.Input(shape=input_shape),

        # ========== PRIMEIRA CONVOLUÇÃO ==========
        # Conv2D: extrai features locais usando 32 filtros 3x3
        # Padding='same': mantém dimensões espaciais
        # ReLU: ativação não-linear
        layers.Conv2D(
            filters=conv1_filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            name='conv2d_1'
        ),

        # ========== PRIMEIRO POOLING ==========
        # MaxPooling2D: reduz dimensionalidade mantendo features mais relevantes
        # Pool size (2,2) reduz dimensões pela metade
        layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            name='maxpooling2d_1'
        ),

        # ========== SEGUNDA CONVOLUÇÃO ==========
        # Conv2D: extrai features mais complexas usando 64 filtros
        # Trabalha com representação reduzida do pooling anterior
        layers.Conv2D(
            filters=conv2_filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            name='conv2d_2'
        ),

        # ========== SEGUNDO POOLING ==========
        # MaxPooling2D: redução dimensional adicional
        layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            name='maxpooling2d_2'
        ),

        # ========== FLATTEN ==========
        # Converte matriz 3D em vetor 1D para camadas Dense
        layers.Flatten(
            name='flatten'
        ),

        # ========== CAMADA DENSA OCULTA ==========
        # Camada fully connected para aprender representações não-lineares
        layers.Dense(
            units=dense_units,
            activation='relu',
            name='dense_hidden'
        ),

        # ========== DROPOUT (REGULARIZAÇÃO) ==========
        # Desativa aleatoriamente neurônios durante treinamento
        # Reduz overfitting melhorando generalização
        layers.Dropout(
            rate=dropout_rate,
            name='dropout'
        ),

        # ========== CAMADA DE SAÍDA ==========
        # Dense final com softmax para probabilidades das classes
        # Se n_classes=2, usar 'sigmoid' é equivalente (alternativa)
        layers.Dense(
            units=n_classes,
            activation='softmax' if n_classes > 2 else 'sigmoid',
            name='output'
        ),
    ])

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
