"""
Testes para o módulo cnn_builder.py

Valida a construção e funcionamento dos modelos CNN.
"""

import pytest
import tensorflow as tf
from src.models.cnn_builder import build_cnn_model, get_model_architecture_summary


class TestCNNBuilder:
    """Suite de testes para construção de modelos CNN."""

    def test_build_basic_cnn_binary(self):
        """Testa construção básica de modelo CNN para classificação binária."""
        model = build_cnn_model(input_shape=(64, 64, 3), n_classes=2)
        
        assert model is not None
        assert isinstance(model, tf.keras.Sequential)
        assert len(model.layers) > 0

    def test_build_cnn_multiclass(self):
        """Testa construção de modelo CNN para classificação multiclasse."""
        model = build_cnn_model(input_shape=(64, 64, 3), n_classes=10)
        
        assert model is not None
        output_layer = model.layers[-1]
        assert output_layer.units == 10

    def test_cnn_architecture_layers(self):
        """Verifica se o modelo contém as camadas esperadas."""
        model = build_cnn_model(input_shape=(64, 64, 3), n_classes=2)
        
        layer_types = [layer.__class__.__name__ for layer in model.layers]
        
        # Verificar presença de camadas principais
        assert 'Conv2D' in layer_types
        assert 'MaxPooling2D' in layer_types
        assert 'Flatten' in layer_types
        assert 'Dense' in layer_types
        assert 'Dropout' in layer_types

    def test_cnn_output_shape_binary(self):
        """Testa se o output shape está correto para classificação binária."""
        model = build_cnn_model(input_shape=(64, 64, 3), n_classes=2)
        
        output_shape = model.output_shape
        assert output_shape == (None, 2)  # Batch, classes

    def test_cnn_output_shape_multiclass(self):
        """Testa se o output shape está correto para multiclasse."""
        model = build_cnn_model(input_shape=(32, 32, 3), n_classes=5)
        
        output_shape = model.output_shape
        assert output_shape == (None, 5)

    def test_custom_filters(self):
        """Testa construção com filtros customizados."""
        model = build_cnn_model(
            input_shape=(64, 64, 3),
            n_classes=2,
            conv1_filters=16,
            conv2_filters=32
        )
        
        assert model is not None
        # Verificar filtros na primeira Conv2D
        conv1 = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][0]
        assert conv1.filters == 16

    def test_custom_kernel_size(self):
        """Testa construção com kernel size customizado."""
        model = build_cnn_model(
            input_shape=(64, 64, 3),
            n_classes=2,
            kernel_size=(5, 5)
        )
        
        assert model is not None
        conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        assert len(conv_layers) == 2
        for conv in conv_layers:
            assert conv.kernel_size == (5, 5)

    def test_custom_dense_units(self):
        """Testa construção com número de unidades Dense customizado."""
        model = build_cnn_model(
            input_shape=(64, 64, 3),
            n_classes=2,
            dense_units=256
        )
        
        dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
        hidden_dense = dense_layers[0]  # Primeira Dense
        assert hidden_dense.units == 256

    def test_dropout_rate(self):
        """Testa se dropout rate está correto."""
        dropout_rate = 0.3
        model = build_cnn_model(
            input_shape=(64, 64, 3),
            n_classes=2,
            dropout_rate=dropout_rate
        )
        
        dropout_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dropout)]
        assert len(dropout_layers) > 0
        assert dropout_layers[0].rate == dropout_rate

    def test_no_dropout(self):
        """Testa construção sem dropout (dropout_rate=0)."""
        model = build_cnn_model(
            input_shape=(64, 64, 3),
            n_classes=2,
            dropout_rate=0.0
        )
        
        dropout_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dropout)]
        assert len(dropout_layers) == 1  # Ainda está no modelo
        assert dropout_layers[0].rate == 0.0

    def test_grayscale_input(self):
        """Testa modelo com imagens em escala de cinza (1 canal)."""
        model = build_cnn_model(input_shape=(64, 64, 1), n_classes=2)
        
        assert model is not None
        input_shape = model.input_shape
        assert input_shape[-1] == 1

    def test_invalid_n_classes(self):
        """Testa erro quando n_classes < 2."""
        with pytest.raises(ValueError):
            build_cnn_model(input_shape=(64, 64, 3), n_classes=1)

    def test_invalid_dropout_rate(self):
        """Testa erro quando dropout_rate inválido."""
        with pytest.raises(ValueError):
            build_cnn_model(
                input_shape=(64, 64, 3),
                n_classes=2,
                dropout_rate=1.5
            )

    def test_model_compilation(self):
        """Verifica se modelo está compilado com otimizador e loss."""
        model = build_cnn_model(input_shape=(64, 64, 3), n_classes=2)
        
        assert model.optimizer is not None
        assert model.loss is not None

    def test_architecture_summary(self):
        """Testa função que retorna resumo da arquitetura."""
        model = build_cnn_model(input_shape=(64, 64, 3), n_classes=2)
        summary = get_model_architecture_summary(model)
        
        assert 'total_params' in summary
        assert 'trainable_params' in summary
        assert 'non_trainable_params' in summary
        assert 'layers_info' in summary
        assert summary['total_params'] > 0

    def test_different_input_sizes(self):
        """Testa modelo com diferentes tamanhos de entrada."""
        sizes = [(32, 32, 3), (64, 64, 3), (128, 128, 3), (256, 256, 1)]
        
        for input_shape in sizes:
            model = build_cnn_model(input_shape=input_shape, n_classes=2)
            assert model is not None
            assert model.input_shape[1:] == input_shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
