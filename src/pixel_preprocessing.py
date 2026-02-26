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
) -> tuple:
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
    tuple
        (df_loadings, important_bands_dict) onde:
        - df_loadings é DataFrame com loadings (bandas × componentes)
        - important_bands_dict é dict com bandas ordenadas por importância
          {'PC1': [('B05', 0.45), ('B07', 0.42), ...], 'PC2': [...], ...}
        
    Notes
    -----
    - Loadings indicam a correlação entre banda original e componente principal
    - Valores próximos de 0 indicam pouca influência
    - Valores negativos indicam correlação negativa (mas ainda importantes!)
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
    
    # Encontrar bandas mais importantes (maior loading absoluto) por PC
    important_bands_dict = {}
    for pc in pc_df_columns:
        # Ordenar bandas por loading absoluto (descendente)
        sorted_bands = df_loadings.loc[pc].abs().sort_values(ascending=False)
        important_bands_dict[pc] = [
            (band, df_loadings.loc[pc, band]) 
            for band in sorted_bands.index
        ]
        
        logger.info(f"\n{pc} - Bandas mais importantes:")
        for i, (band, loading) in enumerate(important_bands_dict[pc][:5]):
            logger.info(f"  {i+1}. {band}: {loading:+.4f}")
    
    # Análise de bandas específicas se fornecidas (para validação)
    if focus_bands is not None:
        logger.info(f"\nValidação de bandas focadas: {focus_bands}")
        for band in focus_bands:
            if band in band_cols:
                max_pc = df_loadings[band].abs().idxmax()
                max_value = df_loadings.loc[max_pc, band]
                logger.info(f"  {band}: Principal em {max_pc} (loading={max_value:+.4f})")
            else:
                logger.warning(f"  {band}: Não encontrada no modelo!")
    
    return df_loadings, important_bands_dict


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
        Lista com nomes das bandas (ex: ['B01', 'B02', ..., 'B14']).
    variance_threshold : float, default=0.95
        Limiar de variância para PCA (0.0 a 1.0).
        Exemplo: 0.95 = manter 95% da variância original
    focus_bands : list, optional
        Bandas específicas para análise detalhada de loadings. Se None, nenhuma
        análise focada é realizada. Exemplos:
        - Terras Raras: ['B04', 'B06']
        - VNIR: ['B01', 'B02', 'B03']
        - TIR: ['B10', 'B11', 'B14']
        
    Returns
    -------
    dict
        Dicionário contendo:
        - 'df_pixels': dados reestruturados (pixel_X → bandas)
        - 'df_standardized': dados padronizados (Z-score)
        - 'df_pca': componentes principais reduzidos
        - 'df_loadings': contribuições de cada banda em cada PC
        - 'important_bands': dict com bandas ordenadas por importância
          Formato: {'PC1': [('B05', 0.45), ('B07', 0.42), ...], 'PC2': [...]}
        - 'scaler': objeto StandardScaler (para novos dados)
        - 'pca_model': objeto PCA fitted (para novos dados)
        
    Examples
    --------
    >>> # Foco em Terras Raras (padrão SWIR)
    >>> result = prepare_pixel_pipeline(
    ...     df, 
    ...     band_names=['B01', 'B02', ..., 'B14'],
    ...     variance_threshold=0.95,
    ...     focus_bands=['B04', 'B06']
    ... )
    >>> df_pca = result['df_pca']  # Use isto no seu modelo!
    >>> loadings = result['df_loadings']  # Para entender contribuições
    
    >>> # Foco em Infrared Termal para vulcanismo
    >>> result = prepare_pixel_pipeline(
    ...     df,
    ...     band_names=['B01', 'B02', ..., 'B14'],
    ...     focus_bands=['B10', 'B11', 'B12', 'B13', 'B14']
    ... )
    
    >>> # Sem foco (análise todas as bandas igualmente)
    >>> result = prepare_pixel_pipeline(df, band_names)
    
    >>> # Descobrir quais bandas são mais importantes
    >>> important = result['important_bands']
    >>> for pc, bands in important.items():
    ...     print(f"{pc}: {bands[:3]}")
    
    >>> # Salvando modelos para reutilizar com novos dados
    >>> import pickle
    >>> pickle.dump(result['scaler'], open('scaler.pkl', 'wb'))
    >>> pickle.dump(result['pca_model'], open('pca_model.pkl', 'wb'))
    """
    logger.info("Iniciando pipeline completo de preparação de pixels...")
    
    # Passo 1: Preparar dados de pixel
    df_pixels = prepare_pixel_data(df, band_names)
    
    # Passo 2: Padronizar bandas
    df_standardized, scaler = standardize_bands(df_pixels, band_names)
    
    # Passo 3: Aplicar PCA
    df_pca, pca_model = apply_pca(df_standardized, band_names, variance_threshold)
    
    # Passo 4: Analisar loadings e encontrar bandas mais importantes
    pc_columns = [f'PC{i+1}' for i in range(pca_model.n_components_)]
    df_loadings, important_bands = analyze_pca_loadings(
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
        'important_bands': important_bands,  # ← NEW! Bandas realmente importantes
        'scaler': scaler,
        'pca_model': pca_model
    }


def prepare_for_neural_network(
    df_pca: pd.DataFrame,
    target_column: str = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Prepara dados PCA para entrada em rede neural (train/test split).
    
    Separa os dados PCA em conjunto de treinamento e validação para redes neurais.
    
    Parameters
    ----------
    df_pca : pd.DataFrame
        DataFrame contendo os componentes principais (ex: saída de apply_pca).
    target_column : str, optional
        Nome da coluna de rótulo/target se houver. Se None, retorna apenas features.
    test_size : float, default=0.2
        Proporção dos dados para teste (0.2 = 20% teste, 80% treino).
    random_state : int, default=42
        Seed para reprodutibilidade na divisão train/test.
        
    Returns
    -------
    dict
        Dicionário contendo:
        - 'X_train': Array de features para treino
        - 'X_test': Array de features para teste
        - 'y_train': Array de targets para treino (se target_column fornecido)
        - 'y_test': Array de targets para teste (se target_column fornecido)
        - 'feature_columns': Nomes das colunas de entrada (PC1, PC2, ...)
        - 'train_indices': Índices das amostras de treino
        - 'test_indices': Índices das amostras de teste
        
    Example
    -------
    >>> from src.pixel_preprocessing import apply_pca, prepare_for_neural_network
    >>> df_pca, pca_model = apply_pca(df_standardized, band_names)
    >>> 
    >>> # Sem target (apenas features PCA)
    >>> dataset = prepare_for_neural_network(df_pca)
    >>> X_train = dataset['X_train']
    >>> 
    >>> # Com target para classificação
    >>> dataset = prepare_for_neural_network(df_pca, target_column='mineral_type')
    >>> X_train, y_train = dataset['X_train'], dataset['y_train']
    >>> 
    >>> # Usar em Keras
    >>> from tensorflow.keras.models import Sequential
    >>> from tensorflow.keras.layers import Dense
    >>> model = Sequential([
    ...     Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    ...     Dense(32, activation='relu'),
    ...     Dense(num_classes, activation='softmax')
    ... ])
    >>> model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)
    """
    from sklearn.model_selection import train_test_split
    
    logger.info(f"Preparando dados para rede neural (test_size={test_size})...")
    
    # Extrair features (componentes principais)
    pc_columns = [col for col in df_pca.columns if col.startswith('PC')]
    X = df_pca[pc_columns].values
    
    feature_columns = pc_columns
    logger.info(f"Features: {feature_columns}")
    
    # Se houver target
    if target_column is not None and target_column in df_pca.columns:
        y = df_pca[target_column].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
        logger.info(f"Dados com target '{target_column}':")
        logger.info(f"  - Train: {X_train.shape[0]} amostras")
        logger.info(f"  - Test: {X_test.shape[0]} amostras")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': feature_columns,
            'feature_names': pc_columns,
            'n_features': len(feature_columns),
            'train_indices': None,
            'test_indices': None
        }
    else:
        # Sem target - apenas features
        X_train, X_test = train_test_split(
            X,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        logger.info(f"Dados sem target (apenas features PCA):")
        logger.info(f"  - Train: {X_train.shape[0]} amostras")
        logger.info(f"  - Test: {X_test.shape[0]} amostras")
        logger.info(f"  - Features (componentes): {len(feature_columns)}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': None,
            'y_test': None,
            'feature_columns': feature_columns,
            'feature_names': pc_columns,
            'n_features': len(feature_columns),
            'train_indices': None,
            'test_indices': None
        }
