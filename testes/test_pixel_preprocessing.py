"""
Testes para o módulo pixel_preprocessing.

Testa as funções principais de preparação de pixels e PCA,
garantindo que funcionam corretamente com dados realistas.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.pixel_preprocessing import (
    prepare_pixel_data,
    standardize_bands,
    apply_pca,
    analyze_pca_loadings,
    prepare_pixel_pipeline
)


@pytest.fixture
def sample_data():
    """Cria um DataFrame de exemplo com dados de pixels."""
    np.random.seed(42)
    
    height, width = 32, 32  # Dimensões do chip
    pixels_per_chip = height * width
    n_bands = 5  # Número de bandas
    n_chips = 10  # Número de chips
    
    data = {
        'height': [height] * n_chips,
        'width': [width] * n_chips,
    }
    
    # Gerar colunas de pixel para cada banda
    band_names = [f'B{i+1:02d}' for i in range(n_bands)]
    for band_idx, band_name in enumerate(band_names):
        for pixel_idx in range(pixels_per_chip):
            col_name = f'pixel_{band_idx * pixels_per_chip + pixel_idx}'
            # Gerar valores realistas de bandas
            data[col_name] = np.random.normal(100 + band_idx*20, 15, n_chips)
    
    df = pd.DataFrame(data)
    return df, band_names


class TestPreparePixelData:
    """Testes para a função prepare_pixel_data."""
    
    def test_basic_preparation(self, sample_data):
        """Testa reestruturação básica de dados."""
        df, band_names = sample_data
        df_pixels = prepare_pixel_data(df, band_names)
        
        # Verificar estrutura
        assert len(df_pixels) == 10 * 32 * 32  # chips * altura * largura
        assert all(band in df_pixels.columns for band in band_names)
        assert 'chip_id' in df_pixels.columns
    
    def test_data_conversion_to_numeric(self, sample_data):
        """Testa conversão para tipo numérico."""
        df, band_names = sample_data
        df_pixels = prepare_pixel_data(df, band_names)
        
        # Verificar tipos de dados
        for band in band_names:
            assert pd.api.types.is_numeric_dtype(df_pixels[band])
    
    def test_nan_handling(self, sample_data):
        """Testa tratamento de valores NaN."""
        df, band_names = sample_data
        
        # Inserir alguns NaN intencionalmente
        pixel_cols = [col for col in df.columns if col.startswith('pixel_')]
        df.iloc[0, df.columns.get_loc(pixel_cols[0])] = np.nan
        
        df_pixels = prepare_pixel_data(df, band_names)
        
        # Verificar que não há NaN após processamento
        assert not df_pixels[band_names].isnull().any().any()


class TestStandardizeBands:
    """Testes para a função standardize_bands."""
    
    def test_standardization(self, sample_data):
        """Testa padronização de bandas."""
        df, band_names = sample_data
        df_pixels = prepare_pixel_data(df, band_names)
        df_std, scaler = standardize_bands(df_pixels, band_names)
        
        # Verificar que a média está próxima de 0 e desvio padrão próximo de 1
        for band in band_names:
            assert abs(df_std[band].mean()) < 1e-10
            # Desvio padrão pode variar ligeiramente por questões numéricas
            assert abs(df_std[band].std() - 1.0) < 0.01
    
    def test_scaler_returned(self, sample_data):
        """Testa que o scaler é retornado corretamente."""
        df, band_names = sample_data
        df_pixels = prepare_pixel_data(df, band_names)
        df_std, scaler = standardize_bands(df_pixels, band_names)
        
        assert isinstance(scaler, StandardScaler)
        assert scaler.n_features_in_ == len(band_names)


class TestApplyPCA:
    """Testes para a função apply_pca."""
    
    def test_pca_dimensionality_reduction(self, sample_data):
        """Testa redução de dimensionalidade por PCA."""
        df, band_names = sample_data
        df_pixels = prepare_pixel_data(df, band_names)
        df_std, _ = standardize_bands(df_pixels, band_names)
        
        df_pca, pca_model = apply_pca(df_std, band_names, variance_threshold=0.90)
        
        # Número de componentes deve ser <= bandas originais
        # (pode ser igual se dados pequenos demais para reduzir efetivamente)
        assert df_pca.shape[1] <= len(band_names)
        assert isinstance(pca_model, PCA)
    
    def test_variance_threshold(self, sample_data):
        """Testa se o threshold de variância é respeitado."""
        df, band_names = sample_data
        df_pixels = prepare_pixel_data(df, band_names)
        df_std, _ = standardize_bands(df_pixels, band_names)
        
        variance_threshold = 0.95
        df_pca, pca_model = apply_pca(df_std, band_names, variance_threshold=variance_threshold)
        
        # Verificar que variância explicada atinge o threshold
        cum_var = np.cumsum(pca_model.explained_variance_ratio_)
        assert cum_var[-1] >= variance_threshold
    
    def test_pca_output_shape(self, sample_data):
        """Testa forma do output do PCA."""
        df, band_names = sample_data
        df_pixels = prepare_pixel_data(df, band_names)
        df_std, _ = standardize_bands(df_pixels, band_names)
        
        df_pca, _ = apply_pca(df_std, band_names)
        
        # Número de linhas deve ser igual ao input
        assert df_pca.shape[0] == df_std.shape[0]
        # Colunas devem começar com 'PC'
        for col in df_pca.columns:
            assert col.startswith('PC')


class TestAnalyzePCALoadings:
    """Testes para a função analyze_pca_loadings."""
    
    def test_loadings_extraction(self, sample_data):
        """Testa extração de loadings."""
        df, band_names = sample_data
        df_pixels = prepare_pixel_data(df, band_names)
        df_std, _ = standardize_bands(df_pixels, band_names)
        
        df_pca, pca_model = apply_pca(df_std, band_names)
        pc_columns = [f'PC{i+1}' for i in range(pca_model.n_components_)]
        
        df_loadings, important_bands = analyze_pca_loadings(pca_model, band_names, pc_columns)
        
        # Verificar shape dos loadings
        assert df_loadings.shape[0] == len(pc_columns)
        assert df_loadings.shape[1] == len(band_names)
        # Verificar important_bands
        assert isinstance(important_bands, dict)
        assert all(pc in important_bands for pc in pc_columns)
    
    def test_focus_bands_analysis(self, sample_data):
        """Testa análise de bandas focadas."""
        df, band_names = sample_data
        df_pixels = prepare_pixel_data(df, band_names)
        df_std, _ = standardize_bands(df_pixels, band_names)
        
        df_pca, pca_model = apply_pca(df_std, band_names)
        pc_columns = [f'PC{i+1}' for i in range(pca_model.n_components_)]
        
        focus_bands = [band_names[0], band_names[1]]
        df_loadings, important_bands = analyze_pca_loadings(
            pca_model, band_names, pc_columns, focus_bands=focus_bands
        )
        
        # Deve retornar os loadings mesmo com focus_bands
        assert all(band in df_loadings.columns for band in focus_bands)
        # Verificar important_bands
        assert isinstance(important_bands, dict)
        assert len(important_bands) == len(pc_columns)


class TestPixelPipeline:
    """Testes para a função prepare_pixel_pipeline."""
    
    def test_pipeline_complete(self, sample_data):
        """Testa execução completa do pipeline."""
        df, band_names = sample_data
        result = prepare_pixel_pipeline(df, band_names)
        
        # Verificar que todos os outputs estão presentes
        assert 'df_pixels' in result
        assert 'df_standardized' in result
        assert 'df_pca' in result
        assert 'df_loadings' in result
        assert 'scaler' in result
        assert 'pca_model' in result
    
    def test_pipeline_with_focus_bands(self, sample_data):
        """Testa pipeline com bandas focadas."""
        df, band_names = sample_data
        focus_bands = [band_names[0], band_names[-1]]
        
        result = prepare_pixel_pipeline(
            df, band_names, focus_bands=focus_bands
        )
        
        assert 'df_loadings' in result
        assert all(band in result['df_loadings'].columns for band in focus_bands)
    
    def test_pipeline_consistency(self, sample_data):
        """Testa consistência entre componentes do pipeline."""
        df, band_names = sample_data
        result = prepare_pixel_pipeline(df, band_names)
        
        # df_standardized e df_pca devem ter mesmo número de linhas
        assert result['df_standardized'].shape[0] == result['df_pca'].shape[0]
        
        # Número de componentes do modelo deve bater com DataFrame PCA
        assert result['pca_model'].n_components_ == result['df_pca'].shape[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
