import os

def encontrar_arquivo_aster(numero_amostra, aster_source_dir):
    """
    Localiza o caminho esperado para o arquivo multibanda ASTER de uma amostra.

    Esta função assume uma estrutura de diretório onde cada amostra tem uma pasta
    nomeada por seu numero_amostra, e dentro dela está o arquivo 'chip_2000m_multiband.tif'.

    Parameters
    ----------
    numero_amostra : int or str
        Identificador único da amostra.
    aster_source_dir : str
        Caminho para o diretório raiz contendo as pastas das imagens ASTER (e.g., '/content/drive/MyDrive/M9/Dados/ASTER_IMG/').

    Returns
    -------
    str or None
        Caminho completo para o arquivo 'chip_2000m_multiband.tif' se existir,
        ou None caso contrário.
    """
    # Constrói o caminho completo para o arquivo esperado
    image_path = os.path.join(aster_source_dir, str(numero_amostra), 'chip_2000m_multiband.tif')

    # Verifica se o arquivo existe
    if os.path.exists(image_path):
        return image_path
    else:
        print(f"Erro: Arquivo de imagem não encontrado para a amostra {numero_amostra} em {image_path}.")
        return None