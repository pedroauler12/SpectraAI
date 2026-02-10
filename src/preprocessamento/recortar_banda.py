import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from .encontrar_arquivo_aster import encontrar_arquivo_aster

def recortar_banda_da_amostra(numero_amostra, band_index, df, aster_source_dir):
    """
    Extrai um chip de 128x128 pixels de uma imagem ASTER para uma amostra específica.
    
    Função que recebe o número de uma amostra e o índice de uma banda, localiza
    as coordenadas geográficas dessa amostra, encontra a imagem ASTER correspondente,
    e extrai um recorte quadrado de 128x128 pixels ao redor do ponto. Se o recorte
    ultrapassar as bordas da imagem original, as áreas faltantes são preenchidas com zeros.
    
    Parameters
    ----------
    numero_amostra : int or str
        Identificador único da amostra no DataFrame.
    band_index : int
        Índice da banda ASTER a extrair (0-indexed).
        Bandas ASTER típicas: 0-8 (9 bandas do infravermelho)
    df : pd.DataFrame
        DataFrame contendo as informações das amostras com colunas:
        - 'numero_amostra': identificador da amostra
        - 'latitude_wgs84_decimal': coordenada de latitude
        - 'longitude_wgs84_decimal': coordenada de longitude
    aster_source_dir : str
        Caminho para o diretório contendo as imagens ASTER.
        Espera-se estrutura: {aster_source_dir}/{numero_amostra}/chip_2000m_multiband.tif
    
    Returns
    -------
    np.ndarray
        Array NumPy bidimensional com shape (128, 128) contendo os valores
        dos pixels para a banda específica. Retorna None em caso de erro.
    
    Examples
    --------
    >>> band_chip = recortar_banda_da_amostra(
    ...     numero_amostra=1001,
    ...     band_index=0,
    ...     df=df_amostras,
    ...     aster_source_dir='/path/to/aster/images/'
    ... )
    >>> print(band_chip.shape)  # (128, 128)
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(band_chip, cmap='gray')
    >>> plt.show()
    
    Notes
    -----
    - O tamanho esperado do chip é 128x128 pixels
    - Se o arquivo ASTER não for encontrado, a função retorna None
    - Se a banda não existir na imagem, a função retorna None
    """

    chip_dimension = 128
    half_chip = chip_dimension // 2

    # Localizar a latitude_wgs84_decimal e longitude_wgs84_decimal
    sample_info_df = df[df['numero_amostra'].astype(str) == str(numero_amostra)]

    if sample_info_df.empty:
        print(f"Erro: Amostra {numero_amostra} não encontrada no DataFrame.")
        return None

    latitude = sample_info_df['latitude_wgs84_decimal'].iloc[0]
    longitude = sample_info_df['longitude_wgs84_decimal'].iloc[0]

    try:
        # Encontrar a imagem ASTER correspondente
        aster_image_path = encontrar_arquivo_aster(numero_amostra, aster_source_dir)

        if aster_image_path is None:
            print(f"Aviso: Imagem ASTER não encontrada para amostra {numero_amostra}")
            return None

        # Abrir a imagem ASTER e extrair a banda
        with rasterio.open(aster_image_path) as src:
            # Validar se a banda existe
            if band_index + 1 > src.count:
                print(f"Aviso: Banda {band_index + 1} não existe (imagem tem {src.count} bandas)")
                return None

            # Convert the longitude_wgs84_decimal and latitude_wgs84_decimal into pixel row and column coordinates
            col_orig, row_orig = src.index(longitude, latitude)

            # Calcular as coordenadas do canto superior esquerdo da janela de extração
            target_row_start = row_orig - half_chip
            target_col_start = col_orig - half_chip

            # Assegurar que window_min_row e window_min_col estejam dentro dos limites da imagem
            window_min_row = max(0, target_row_start)
            if src.height > chip_dimension:
                window_min_row = min(window_min_row, src.height - chip_dimension)
            else:
                window_min_row = 0 # Se a imagem for menor que o chip, começa do 0

            window_min_col = max(0, target_col_start)
            if src.width > chip_dimension:
                window_min_col = min(window_min_col, src.width - chip_dimension)
            else:
                window_min_col = 0 # Se a imagem for menor que o chip, começa do 0
            
            # Recalcula as dimensões reais do chip caso a imagem seja menor que a dimensão do chip
            actual_chip_height = min(chip_dimension, src.height - window_min_row)
            actual_chip_width = min(chip_dimension, src.width - window_min_col)

            if actual_chip_height <= 0 or actual_chip_width <= 0:
                print(f"Aviso: Dimensões inválidas do chip para a amostra {numero_amostra}. Pulando.")
                return None

            # Criar um objeto rasterio.windows.Window
            window = Window(window_min_col, window_min_row, actual_chip_width, actual_chip_height)

            # Ler os dados da banda específica (rasterio usa indexação base 1 para bandas)
            image_band_chip = src.read(band_index + 1, window=window)
            
            # Preenche com zeros se o chip extraído for menor que chip_dimension
            padded_chip = np.zeros((chip_dimension, chip_dimension), dtype=image_band_chip.dtype)
            padded_chip[:image_band_chip.shape[0], :image_band_chip.shape[1]] = image_band_chip

            return padded_chip.astype(np.float32)

    except Exception as e:
        print(f"Erro ao processar amostra {numero_amostra}: {str(e)}")
        return None