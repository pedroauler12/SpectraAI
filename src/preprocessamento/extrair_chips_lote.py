"""
Módulo para extração em lote de chips de imagens ASTER.

Este módulo fornece funcionalidades para extrair múltiplos chips
de imagens ASTER de forma eficiente.
"""

from .recortar_banda import recortar_banda_da_amostra

def extrair_chips_multiplas_amostras(numeros_amostras, band_index, df,
                                     aster_source_dir, verbose=True):
    """
    Extrai chips de múltiplas amostras em lote.
    
    Parameters
    ----------
    numeros_amostras : list or np.ndarray
        Lista de números de amostras para extrair
    band_index : int
        Índice da banda a extrair
    df : pd.DataFrame
        DataFrame com informações das amostras (necessário para buscar coordenadas).
    aster_source_dir : str
        Caminho para o diretório de imagens ASTER
    verbose : bool, optional
        Se True, imprime progresso (padrão: True)
    
    Returns
    -------
    dict
        Dicionário com mapeamento numero_amostra -> array de chip
        Amostras com erro não são incluídas no dicionário
    
    Examples
    --------
    >>> amostras = [1001, 1002, 1003, 1004, 1005]
    >>> chips_dict = extrair_chips_multiplas_amostras(
    ...     numeros_amostras=amostras,
    ...     band_index=0,
    ...     df=df_amostras,
    ...     aster_source_dir='/path/to/aster/'
    ... )
    >>> print(f"Chips extraídos: {len(chips_dict)}")
    >>> chip = chips_dict[1001]  # Acessar um chip específico
    >>> print(chip.shape)  # (128, 128)
    
    Notes
    -----
    - Amostras que geram erro não são incluídas no dicionário de retorno
    - Se verbose=True, imprime progresso a cada amostra processada
    - Retorna apenas chips extraídos com sucesso
    """
    
    chips = {}
    total = len(numeros_amostras)
    
    for i, numero_amostra in enumerate(numeros_amostras):
        if verbose:
            print(f"Processando amostra {i+1}/{total}: {numero_amostra}", end='\r')
        
        chip = recortar_banda_da_amostra(numero_amostra, band_index, df, aster_source_dir)
        
        if chip is not None:
            chips[numero_amostra] = chip
    
    if verbose:
        print(f"\nChips extraídos com sucesso: {len(chips)}/{total}")
    
    return chips