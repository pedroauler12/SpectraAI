# -*- coding: utf-8 -*-
"""
Módulo para carregamento e processamento de imagens ASTER de 128x128 pixels.

Responsável por:
- Carregar bandas ASTER individuais (VNIR, SWIR, TIR)
- Normalizar/processar imagens
- Reamostragem para 128x128 se necessário
- Aplicar filtros e combinar bandas
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AsterImageLoader:
    """Carregador de imagens ASTER com suporte a diferentes subsistemas."""

    # Mapeamento de bandas para intervalos espectrais (µm)
    BAND_INFO = {
        'B01': {'subsistema': 'VNIR', 'faixa': 'Verde (0.52-0.60)'},
        'B02': {'subsistema': 'VNIR', 'faixa': 'Vermelho (0.63-0.69)'},
        'B03N': {'subsistema': 'VNIR', 'faixa': 'NIR (0.76-0.86)'},
        'B04': {'subsistema': 'SWIR', 'faixa': '1.600-1.700'},
        'B05': {'subsistema': 'SWIR', 'faixa': '2.145-2.185 (Alunita)'},
        'B06': {'subsistema': 'SWIR', 'faixa': '2.185-2.225 (Argilas Al-OH)'},
        'B07': {'subsistema': 'SWIR', 'faixa': '2.235-2.285 (Muscovita)'},
        'B08': {'subsistema': 'SWIR', 'faixa': '2.295-2.365 (Mg-OH)'},
        'B09': {'subsistema': 'SWIR', 'faixa': '2.360-2.430 (Talco/Clorita)'},
        'B10': {'subsistema': 'TIR', 'faixa': '8.125-8.475'},
        'B11': {'subsistema': 'TIR', 'faixa': '8.475-8.825'},
        'B12': {'subsistema': 'TIR', 'faixa': '8.925-9.275'},
        'B13': {'subsistema': 'TIR', 'faixa': '10.25-10.95'},
        'B14': {'subsistema': 'TIR', 'faixa': '10.95-11.65'},
    }

    def __init__(self, amostras_root: Path):
        """
        Inicializa o carregador.

        Parameters
        ----------
        amostras_root : Path
            Raiz do diretório contendo as amostras (ex.: G:\...\ASTER_IMG)
        """
        self.amostras_root = Path(amostras_root)
        if not self.amostras_root.exists():
            raise ValueError(f"Diretório de amostras não existe: {self.amostras_root}")

    def encontrar_arquivo_banda(self, amostra_id: str, banda: str) -> Optional[Path]:
        """
        Localiza o arquivo de uma banda específica na pasta da amostra.

        Parameters
        ----------
        amostra_id : str
            ID da amostra (número ou nome da pasta)
        banda : str
            Identificação da banda (ex.: 'B01', 'B06', 'B13')

        Returns
        -------
        Path or None
            Caminho do arquivo se encontrado, None caso contrário
        """
        amostra_dir = self.amostras_root / amostra_id
        if not amostra_dir.exists():
            logger.warning(f"Pasta da amostra não encontrada: {amostra_dir}")
            return None

        # Procura por arquivos que contenham o identificador da banda
        patterns = [
            f"*{banda}.tif",
            f"*_{banda}.tif",
            f"*_B{banda[-2:]}.tif",
        ]

        for pattern in patterns:
            matches = list(amostra_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def carregar_banda(
        self,
        amostra_id: str,
        banda: str,
        target_shape: int = 128,
        normalize: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Carrega uma banda ASTER e redimensiona para 128x128.

        Parameters
        ----------
        amostra_id : str
            ID da amostra
        banda : str
            Identificação da banda
        target_shape : int
            Tamanho alvo para a imagem (padrão: 128)
        normalize : bool
            Se True, normaliza a imagem para [0, 1]

        Returns
        -------
        np.ndarray or None
            Array 2D da imagem, ou None se não encontrada/carregada
        """
        arquivo = self.encontrar_arquivo_banda(amostra_id, banda)
        if arquivo is None:
            logger.warning(f"Banda {banda} não encontrada para amostra {amostra_id}")
            return None

        try:
            with rasterio.open(arquivo) as src:
                imagem = src.read(1).astype(np.float32)
        except Exception as e:
            logger.error(f"Erro ao carregar {arquivo}: {e}")
            return None

        # Redimensiona se necessário usando cv2
        if imagem.shape != (target_shape, target_shape):
            import cv2

            imagem = cv2.resize(imagem, (target_shape, target_shape), interpolation=cv2.INTER_CUBIC)

        # Normaliza
        if normalize:
            vmin, vmax = np.nanpercentile(imagem, [2, 98])
            if vmax > vmin:
                imagem = (imagem - vmin) / (vmax - vmin)
                imagem = np.clip(imagem, 0, 1)
            else:
                imagem = np.zeros_like(imagem)

        return imagem

    def carregar_multiplas_bandas(
        self,
        amostra_id: str,
        bandas: List[str],
        target_shape: int = 128,
        normalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Carrega múltiplas bandas de uma amostra.

        Parameters
        ----------
        amostra_id : str
            ID da amostra
        bandas : List[str]
            Lista de bandas a carregar (ex.: ['B01', 'B02', 'B03N'])
        target_shape : int
            Tamanho alvo para as imagens
        normalize : bool
            Se True, normaliza cada imagem para [0, 1]

        Returns
        -------
        Dict[str, np.ndarray]
            Dicionário {banda: array_2d}
        """
        resultado = {}
        for banda in bandas:
            imagem = self.carregar_banda(amostra_id, banda, target_shape, normalize)
            if imagem is not None:
                resultado[banda] = imagem
        return resultado

    def criar_rgb_falsa_cor(
        self,
        amostra_id: str,
        banda_r: str = 'B06',
        banda_g: str = 'B05',
        banda_b: str = 'B02',
        target_shape: int = 128,
    ) -> Optional[np.ndarray]:
        """
        Cria uma composição RGB falsa-cor para visualização.

        Common compositions:
        - SWIR RGB: B06, B05, B02 (para argilas/alteração mineral)
        - VNIR RGB: B03N, B02, B01 (composição natural)
        - Índice mineral: B06/(B05+B04) em pseudo-cor

        Parameters
        ----------
        amostra_id : str
            ID da amostra
        banda_r : str
            Banda para canal Red
        banda_g : str
            Banda para canal Green
        banda_b : str
            Banda para canal Blue
        target_shape : int
            Tamanho alvo

        Returns
        -------
        np.ndarray or None
            Array RGB de shape (128, 128, 3)
        """
        bandas = {banda_r, banda_g, banda_b}
        carregadas = self.carregar_multiplas_bandas(
            amostra_id, list(bandas), target_shape, normalize=True
        )

        if len(carregadas) < 3:
            logger.warning(f"Não foi possível carregar todas as 3 bandas para {amostra_id}")
            return None

        rgb = np.stack([carregadas[banda_r], carregadas[banda_g], carregadas[banda_b]], axis=2)
        return np.clip(rgb, 0, 1)

    def aplicar_filtro_gaussiano(
        self, imagem: np.ndarray, sigma: float = 1.0
    ) -> np.ndarray:
        """
        Aplica filtro Gaussiano à imagem.

        Parameters
        ----------
        imagem : np.ndarray
            Imagem de entrada
        sigma : float
            Desvio padrão do kernel Gaussiano

        Returns
        -------
        np.ndarray
            Imagem filtrada
        """
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(imagem, sigma=sigma)

    def calcular_indice_mineral(
        self,
        amostra_id: str,
        banda_num: str = 'B06',
        banda_den1: str = 'B05',
        banda_den2: str = 'B04',
        target_shape: int = 128,
    ) -> Optional[np.ndarray]:
        """
        Calcula índices minerais como razões de bandas.

        Exemplo: Índice de Argilas = B06 / (B05 + B04)
                 Índice de Carbonatos = B13 / B14

        Parameters
        ----------
        amostra_id : str
            ID da amostra
        banda_num : str
            Banda do numerador
        banda_den1 : str
            Primeira banda do denominador
        banda_den2 : str
            Segunda banda do denominador (opcional)
        target_shape : int
            Tamanho alvo

        Returns
        -------
        np.ndarray or None
            Índice mineral como imagem 2D
        """
        bandas_needed = {banda_num, banda_den1}
        if banda_den2:
            bandas_needed.add(banda_den2)

        carregadas = self.carregar_multiplas_bandas(
            amostra_id, list(bandas_needed), target_shape, normalize=True
        )

        if banda_num not in carregadas or banda_den1 not in carregadas:
            logger.warning(f"Bandas necessárias não encontradas para amostra {amostra_id}")
            return None

        numerador = carregadas[banda_num]
        denominador = carregadas[banda_den1]

        if banda_den2 and banda_den2 in carregadas:
            denominador = denominador + carregadas[banda_den2]

        # Evita divisão por zero
        resultado = np.divide(
            numerador, denominador, where=denominador > 1e-6, out=np.zeros_like(numerador)
        )

        # Normaliza para [0, 1]
        vmin, vmax = np.nanpercentile(resultado, [2, 98])
        if vmax > vmin:
            resultado = (resultado - vmin) / (vmax - vmin)
            resultado = np.clip(resultado, 0, 1)

        return resultado

    @staticmethod
    def listar_amostras_disponiveis(amostras_root: Path) -> List[str]:
        """
        Lista os IDs de todas as amostras disponíveis no diretório.

        Parameters
        ----------
        amostras_root : Path
            Raiz do diretório de amostras

        Returns
        -------
        List[str]
            Lista de IDs de amostras (nomes das pastas)
        """
        amostras_root = Path(amostras_root)
        amostras = [d.name for d in amostras_root.iterdir() if d.is_dir()]
        return sorted(amostras)
