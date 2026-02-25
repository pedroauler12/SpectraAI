"""
Módulo de préprocessamento de pixels para preparação de entrada do modelo.

Este módulo contém funções para:
- Reestruturação de dados de pixels de formato tabular
- Padronização de bandas espectrais
- Aplicação de PCA para redução de dimensionalidade
- Análise de loadings dos componentes principais

Exemplo de uso:
    >>> from src.pixel_preprocessing import prepare_pixel_data, standardize_bands, apply_pca
    >>> df_pixels = prepare_pixel_data(df, band_names=['B1', 'B2', 'B3', 'B4'])
    >>> df_standardized = standardize_bands(df_pixels, ['B1', 'B2', 'B3', 'B4'])
    >>> df_pca, pca_model = apply_pca(df_standardized, ['B1', 'B2', 'B3', 'B4'])
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def prepare_pixel_data(df: pd.DataFrame, band_names: list) -> pd.DataFrame:
    """
    Reestrutura o DataFrame, convertendo colunas pixel_X em bandas e tratando tipos.
    
    Transforma dados de pixels em formato de coluna única para um formato 
    estruturado com bandas espectrais como colunas. Também realiza tratamento 
    de tipos de dados, convertendo para numérico e preenchendo valores ausentes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo colunas 'height', 'width' e colunas 'pixel_*'
        que serão reestruturadas em bandas.
    band_names : list
        Lista com os nomes das bandas espectrais (ex: ['B01', 'B02', 'B03', ...])
        
    Returns
    -------
    pd.DataFrame
        DataFrame reestruturado com colunas de banda e coluna 'chip_id'
        
    Notes
    -----
    - Espera que o número de colunas pixel_* corresponda a len(band_names) * altura * largura
    - Valores NaN após conversão numérica são preenchidos com 0
    - Não remove colunas originais de pixel_*
    
    Raises
    ------
    ValueError
        Se as dimensões não corresponderem ao esperado
    """
    logger.info("Iniciando preparação de dados de pixel...")
    
    pixel_cols = [col for col in df.columns if col.startswith('pixel_')]
    num_bands = len(band_names)
    
    # Detectar dimensões
    if 'height' in df.columns and 'width' in df.columns:
        height = df['height'].iloc[0]
        width = df['width'].iloc[0]
        pixels_per_chip = height * width
        logger.info(f"Dimensões encontradas: {height} x {width}")
    else:
        # Detectar automaticamente baseado no número de pixel columns
        pixels_per_chip = len(pixel_cols) // num_bands
        side_len = int(np.sqrt(pixels_per_chip))
        if side_len * side_len != pixels_per_chip:
            raise ValueError(
                f"Não conseguiu determinar dimensões do chip automaticamente. "
                f"Pixels por banda: {pixels_per_chip} (não é quadrado perfeito). "
                f"Add 'height' e 'width' columns no DataFrame."
            )
        height = width = side_len
        logger.info(f"Dimensões detectadas automaticamente: {height} x {width}")
    
    expected_pixels = num_bands * pixels_per_chip
    
    if len(pixel_cols) != expected_pixels:
        logger.warning(
            f"Número de colunas de pixel ({len(pixel_cols)}) não corresponde ao "
            f"esperado ({expected_pixels}). Processando ainda assim..."
        )
    
    logger.debug(f"Pixels por chip: {pixels_per_chip}")
    logger.debug(f"Colunas de pixel encontradas: {len(pixel_cols)}")
    
    # Organizar colunas de pixel por banda
    band_col_ranges = {}
    for i, band_name in enumerate(band_names):
        start_col_idx = i * pixels_per_chip
        end_col_idx = (i + 1) * pixels_per_chip
        band_col_ranges[band_name] = pixel_cols[start_col_idx:end_col_idx]
    
    # Reestruturar em novo DataFrame
    df_pixels_list = []
    for chip_id, row in df.iterrows():
        # Garantir que todos os arrays têm o mesmo tamanho preenchendo NaNs
        chip_data = {}
        for band_name, cols in band_col_ranges.items():
            values = row[cols].values
            # Se tem NaN, preencher com 0 (comum em datasets incompletos)
            if len(values) < pixels_per_chip:
                values = np.pad(values, (0, pixels_per_chip - len(values)), 
                              constant_values=np.nan)
            chip_data[band_name] = values[:pixels_per_chip]
        
        # Criar DataFrame com os dados garantidamente do mesmo tamanho
        chip_pixels_df = pd.DataFrame(chip_data)
        chip_pixels_df['chip_id'] = chip_id
        df_pixels_list.append(chip_pixels_df)
    
    df_pixels = pd.concat(df_pixels_list, ignore_index=True)
    
    # Converter para tipo numérico
    logger.info("Convertendo colunas de banda para tipo numérico...")
    for col in band_names:
        df_pixels[col] = pd.to_numeric(df_pixels[col], errors='coerce')
    
    # Tratar valores NaN
    if df_pixels[band_names].isnull().any().any():
        num_nans = df_pixels[band_names].isnull().sum().sum()
        logger.warning(f"Encontrados {num_nans} valores NaN; preenchendo com 0.")
        df_pixels[band_names] = df_pixels[band_names].fillna(0)
    
    logger.info(f"Reestruturação completa. DataFrame contém {len(df_pixels)} linhas.")
    return df_pixels


def standardize_bands(df_pixels: pd.DataFrame, band_cols: list) -> tuple:
    """
    Aplica StandardScaler às colunas de banda especificadas.
    
    Realiza padronização (z-score normalization) nas colunas de banda,
    transformando os dados para média 0 e desvio padrão 1.
    
    Parameters
    ----------
    df_pixels : pd.DataFrame
        DataFrame com colunas de banda a padronizar.
    band_cols : list
        Lista com os nomes das colunas de banda a padronizar.
        
    Returns
    -------
    tuple
        (df_standardized, scaler) onde:
        - df_standardized é o DataFrame com bandas padronizadas
        - scaler é o objeto StandardScaler ajustado (para uso posterior)
        
    Notes
    -----
    - Colunas não especificadas em band_cols são mantidas inalteradas
    - O scaler pode ser usado para transformar novos dados
    """
    logger.info("Aplicando StandardScaler às colunas de banda...")
    
    scaler = StandardScaler()
    df_pixels_standardized = df_pixels.copy()
    df_pixels_standardized[band_cols] = scaler.fit_transform(df_pixels[band_cols])
    
    logger.info(f"Padronização completa. {len(band_cols)} bandas transformadas.")
    return df_pixels_standardized, scaler


def apply_pca(
    df_pixels_standardized: pd.DataFrame,
    band_cols: list,
    variance_threshold: float = 0.95
) -> tuple:
    """
    Aplica PCA e seleciona componentes a partir de um limiar de variância.
    
    Realiza análise de componentes principais (PCA) nos dados padronizados
    e retorna um DataFrame com os componentes principais que explicam
    uma percentagem mínima da variância total.
    
    Parameters
    ----------
    df_pixels_standardized : pd.DataFrame
        DataFrame com bandas já padronizadas (ex: saída de standardize_bands).
    band_cols : list
        Lista com os nomes das colunas de banda usadas no PCA.
    variance_threshold : float, default=0.95
        Limiar de variância cumulativa para seleção de componentes.
        Exemplo: 0.95 mantém componentes até explicarem 95% da variância.
        
    Returns
    -------
    tuple
        (df_pca, pca_model) onde:
        - df_pca é DataFrame com os componentes principais
        - pca_model é o modelo PCA ajustado (com atributo components_ para loadings)
        
    Notes
    -----
    - O número de componentes é determinado automaticamente pelo threshold
    - Os nomes das colunas são 'PC1', 'PC2', ...
    """
    logger.info(
        f"Realizando PCA (variância esperada > {variance_threshold*100:.0f}%)..."
    )
    
    # Primeiro PCA para determinar número de componentes
    pca_temp = PCA()
    pca_temp.fit(df_pixels_standardized[band_cols])
    cum_var_ratio = np.cumsum(pca_temp.explained_variance_ratio_)
    
    # Encontrar número ideal de componentes
    optimal_components = np.where(cum_var_ratio >= variance_threshold)[0]
    if len(optimal_components) == 0:
        optimal_components = len(band_cols)  # Usar todos se não alcançar threshold
    else:
        optimal_components = optimal_components[0] + 1
    
    logger.info(f"Componentes selecionados: {optimal_components}")
    logger.debug(f"Variância explicada cumulativa: {cum_var_ratio}")
    
    # PCA final com número otimizado de componentes
    pca_final = PCA(n_components=optimal_components)
    pc_data = pca_final.fit_transform(df_pixels_standardized[band_cols])
    
    pc_columns = [f'PC{i+1}' for i in range(optimal_components)]
    df_pca = pd.DataFrame(data=pc_data, columns=pc_columns)
    
    logger.info(f"Transformação PCA completa com {optimal_components} componentes.")
    return df_pca, pca_final


def analyze_pca_loadings(
    pca_model: PCA,
    band_cols: list,
    pc_df_columns: list = None,
    focus_bands: list = None
) -> pd.DataFrame:
    """
    Analisa os loadings dos componentes principais.
    
    Extrai a matriz de loadings do modelo PCA e permite análise detalhada
    da contribuição de cada banda para cada componente principal.
    Útil para interpretar o significado físico dos componentes.
    
    Parameters
    ----------
    pca_model : PCA
        Modelo PCA ajustado (objeto sklearn.decomposition.PCA).
    band_cols : list
        Lista com os nomes das bandas originais.
    pc_df_columns : list, optional
        Nomes das colunas dos componentes principais.
        Se None, gera automaticamente ['PC1', 'PC2', ...].
    focus_bands : list, optional
        Banda(s) específica(s) a analisar (ex: ['B4', 'B6']).
        Se None, analisa todas as bandas.
        
    Returns
    -------
    pd.DataFrame
        DataFrame com loadings (bandas como índice, componentes como colunas).
        
    Notes
    -----
    - Loadings indicam a correlação entre banda original e componente principal
    - Valores próximos de 0 indicam pouca influência
    - Valores negativos indicam correlação negativa
    - Para mineralogia de Terras Raras, B4 e B6 são frequentemente analisados
    """
    logger.info("Analisando loadings dos componentes principais...")
    
    if pc_df_columns is None:
        pc_df_columns = [f'PC{i+1}' for i in range(len(pca_model.components_))]
    
    # Construir DataFrame de loadings
    loadings = pca_model.components_
    df_loadings = pd.DataFrame(
        data=loadings.T,
        index=band_cols,
        columns=pc_df_columns
    ).T
    
    logger.debug(f"Loadings calculados: {df_loadings.shape}")
    
    # Análise de bandas específicas se fornecidas
    if focus_bands is not None:
        logger.info(f"Analisando loadings focados em: {focus_bands}")
        for pc in pc_df_columns:
            for band in focus_bands:
                if band in df_loadings.columns:
                    loading_value = df_loadings.loc[pc, band]
                    logger.debug(f"  {pc} - {band}: {loading_value:.4f}")
        
        # Encontrar componentes com maior influência de bandas focadas
        for band in focus_bands:
            if band in df_loadings.columns:
                max_pc = df_loadings[band].abs().idxmax()
                max_value = df_loadings.loc[max_pc, band]
                logger.info(
                    f"Maior loading para {band}: {max_pc} ({max_value:.4f})"
                )
    
    return df_loadings


def prepare_pixel_pipeline(
    df: pd.DataFrame,
    band_names: list,
    variance_threshold: float = 0.95,
    focus_bands: list = None
) -> dict:
    """
    Pipeline completo de preparação de pixels com PCA.
    
    Funcão conveniente que executa todos os passos de preparação:
    preparação de dados, padronização, PCA e análise de loadings.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame bruto com colunas pixel_*.
    band_names : list
        Lista com nomes das bandas.
    variance_threshold : float, default=0.95
        Limiar de variância para PCA.
    focus_bands : list, optional
        Bandas para análise detalhada de loadings.
        
    Returns
    -------
    dict
        Dicionário contendo:
        - 'df_pixels': dados reestruturados
        - 'df_standardized': dados padronizados
        - 'df_pca': componentes principais
        - 'df_loadings': loadings do PCA
        - 'scaler': objeto StandardScaler
        - 'pca_model': objeto PCA
        
    Example
    -------
    >>> result = prepare_pixel_pipeline(df, ['B1', 'B2', 'B3'], focus_bands=['B2', 'B3'])
    >>> df_pca = result['df_pca']
    >>> loadings = result['df_loadings']
    """
    logger.info("Iniciando pipeline completo de preparação de pixels...")
    
    # Passo 1: Preparar dados de pixel
    df_pixels = prepare_pixel_data(df, band_names)
    
    # Passo 2: Padronizar bandas
    df_standardized, scaler = standardize_bands(df_pixels, band_names)
    
    # Passo 3: Aplicar PCA
    df_pca, pca_model = apply_pca(df_standardized, band_names, variance_threshold)
    
    # Passo 4: Analisar loadings
    pc_columns = [f'PC{i+1}' for i in range(pca_model.n_components_)]
    df_loadings = analyze_pca_loadings(
        pca_model,
        band_names,
        pc_columns,
        focus_bands
    )
    
    logger.info("Pipeline de preparação de pixels concluído com sucesso.")
    
    return {
        'df_pixels': df_pixels,
        'df_standardized': df_standardized,
        'df_pca': df_pca,
        'df_loadings': df_loadings,
        'scaler': scaler,
        'pca_model': pca_model
    }
