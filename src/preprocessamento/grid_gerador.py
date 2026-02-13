# -*- coding: utf-8 -*-
"""
Módulo para geração de grids analisados de imagens ASTER.

Cria visualizações em grid para análises comparativas:
- Positivos vs Negativos
- Diferentes minerais / características mineralógicas
- Diferentes bandas espectrais
- Aplicação de filtros e combinações de bandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging

from carregador_aster import AsterImageLoader

logger = logging.getLogger(__name__)


class GridGerador:
    """Gerador de grids analisados para exploração visual de imagens ASTER."""

    def __init__(self, carregador: AsterImageLoader, excel_path: str):
        """
        Inicializa o gerador de grids.

        Parameters
        ----------
        carregador : AsterImageLoader
            Instância do carregador de imagens ASTER
        excel_path : str
            Caminho do arquivo Excel com base de dados positiva/negativa
        """
        self.carregador = carregador
        self.df_banco = pd.read_excel(excel_path)
        self._processar_banco()

    def _processar_banco(self):
        """Processa o DataFrame do banco de dados para facilitar consultas."""
        # Cria dicionário para mapeamento de colunas (original -> nome padrão)
        colnames_lower = {col.lower().strip(): col for col in self.df_banco.columns}
        
        # Tenta encontrar coluna de ID
        self.col_id = None
        for pattern in ['numero_amostra', 'numero', 'id_amostra', 'amostra_id', 'sample_id', 'id']:
            if pattern in colnames_lower:
                self.col_id = colnames_lower[pattern]
                break
        
        if self.col_id is None:
            # Tenta usar a primeira coluna como ID
            self.col_id = self.df_banco.columns[0]
        
        # Tenta encontrar coluna de rótulo
        self.col_label = None
        for pattern in ['classe_balanceamento', 'classe', 'label', 'positivo_negativo', 'resultado', 'class', 'target', 'y']:
            if pattern in colnames_lower:
                self.col_label = colnames_lower[pattern]
                break
        
        # Tenta encontrar coluna de mineral/litologia
        self.col_mineral = None
        for pattern in ['litologia_padronizada', 'mineralog', 'mineral', 'tipo', 'litologia']:
            if pattern in colnames_lower:
                self.col_mineral = colnames_lower[pattern]
                break

        logger.info(f"Banco processado: {len(self.df_banco)} amostras")
        logger.info(f"Coluna ID: {self.col_id}, Label: {self.col_label}, Mineral: {self.col_mineral}")

    def obter_amostras_por_classe(
        self, positivo: bool = True
    ) -> List[str]:
        """
        Obtém lista de amostras de uma classe específica.

        Parameters
        ----------
        positivo : bool
            Se True, retorna positivos; se False, retorna negativos

        Returns
        -------
        List[str]
            IDs das amostras
        """
        if self.col_label is None:
            logger.warning("Coluna de rótulo não encontrada")
            return []

        df = self.df_banco.copy()
        
        # Normaliza valores de label para uppercase
        if df[self.col_label].dtype == object:
            df['label_norm'] = df[self.col_label].astype(str).str.upper().str.strip()
        else:
            df['label_norm'] = df[self.col_label].astype(str)

        # Define padrões para positivo
        if positivo:
            # Padrões para POSITIVO
            positivos_mask = df['label_norm'].isin([
                'POSITIVO', 'TRUE', '1', 'SIM', 'YES', 'POSITIVE', 'POS'
            ]) | (df['label_norm'] == '1')
            amostras = df[positivos_mask][self.col_id].astype(str).tolist()
        else:
            # Padrões para NEGATIVO
            positivos_mask = df['label_norm'].isin([
                'POSITIVO', 'TRUE', '1', 'SIM', 'YES', 'POSITIVE', 'POS'
            ]) | (df['label_norm'] == '1')
            amostras = df[~positivos_mask][self.col_id].astype(str).tolist()

        return amostras

    def obter_amostras_por_mineral(self, tipo_mineral: str) -> List[str]:
        """
        Obtém amostras de um tipo mineral específico.

        Parameters
        ----------
        tipo_mineral : str
            Tipo de mineral a filtrar

        Returns
        -------
        List[str]
            IDs das amostras
        """
        if self.col_mineral is None:
            logger.warning("Coluna de mineral não encontrada")
            return []

        df = self.df_banco.copy()
        df['mineral_norm'] = df[self.col_mineral].str.lower().str.strip()

        amostras = df[df['mineral_norm'].str.contains(tipo_mineral.lower(), na=False)][
            self.col_id
        ].astype(str).tolist()

        return amostras

    def filtrar_amostras_disponiveis(self, amostras: List[str]) -> List[str]:
        """
        Filtra apenas amostras que têm imagens disponíveis no carregador.

        Parameters
        ----------
        amostras : List[str]
            Lista de IDs de amostras

        Returns
        -------
        List[str]
            Amostras que têm imagens disponíveis
        """
        disponiveis = []
        for amostra_id in amostras:
            # Verifica se o arquivo exists na raiz do carregador
            amostra_path = self.carregador.amostras_root / amostra_id
            if amostra_path.exists():
                disponiveis.append(amostra_id)
        return disponiveis

    def criar_grid_positivos_vs_negativos(
        self,
        banda: str = 'B06',
        n_amostras: int = 12,
        figsize: Tuple[int, int] = (20, 20),
        title: str = "Positivos vs Negativos",
    ) -> plt.Figure:
        """
        Cria grid comparando imagens positivas e negativas da mesma banda.

        Parameters
        ----------
        banda : str
            Banda espectral a visualizar (ex.: 'B06')
        n_amostras : int
            Número total de amostras (será dividido entre positivos/negativos)
        figsize : Tuple
            Tamanho da figura
        title : str
            Título do gráfico

        Returns
        -------
        plt.Figure
            Figura matplotlib com o grid
        """
        # Obtém amostras por classe
        amostras_pos_all = self.obter_amostras_por_classe(positivo=True)
        amostras_neg_all = self.obter_amostras_por_classe(positivo=False)
        
        # Filtra apenas as disponíveis
        amostras_pos = self.filtrar_amostras_disponiveis(amostras_pos_all)
        amostras_neg = self.filtrar_amostras_disponiveis(amostras_neg_all)
        
        # Limita ao número solicitado
        n_por_classe = n_amostras // 2
        amostras_pos = amostras_pos[:n_por_classe]
        amostras_neg = amostras_neg[:n_por_classe]
        
        n_total = len(amostras_pos) + len(amostras_neg)
        ncols = 6
        nrows = int(np.ceil(n_total / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()

        # Positivos
        for idx, amostra_id in enumerate(amostras_pos):
            imagem = self.carregador.carregar_banda(amostra_id, banda, normalize=True)
            if imagem is not None:
                axes[idx].imshow(imagem, cmap='viridis')
                axes[idx].set_title(f"Positivo #{amostra_id} ({banda})", fontweight='bold', fontsize=9)
            axes[idx].axis('off')

        # Negativos
        offset = len(amostras_pos)
        for idx, amostra_id in enumerate(amostras_neg):
            imagem = self.carregador.carregar_banda(amostra_id, banda, normalize=True)
            if imagem is not None:
                axes[offset + idx].imshow(imagem, cmap='viridis')
                axes[offset + idx].set_title(f"Negativo #{amostra_id} ({banda})", fontweight='bold', fontsize=9)
            axes[offset + idx].axis('off')

        # Oculta eixos vazios
        for idx in range(n_total, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig

    def criar_grid_multiplas_bandas(
        self,
        amostra_id: str,
        bandas: List[str],
        figsize: Tuple[int, int] = (15, 15),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Cria grid mostrando múltiplas bandas da mesma amostra.

        Parameters
        ----------
        amostra_id : str
            ID da amostra
        bandas : List[str]
            Lista de bandas a visualizar (ex.: ['B01', 'B02', 'B06', 'B13'])
        figsize : Tuple
            Tamanho da figura
        title : str, optional
            Título customizado

        Returns
        -------
        plt.Figure
            Figura matplotlib com o grid
        """
        n = len(bandas)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n > 1 else [axes]

        for idx, banda in enumerate(bandas):
            imagem = self.carregador.carregar_banda(amostra_id, banda, normalize=True)
            if imagem is not None:
                im = axes[idx].imshow(imagem, cmap='gray')
                subsistema = self.carregador.BAND_INFO.get(banda, {}).get('subsistema', '')
                axes[idx].set_title(f"{banda} ({subsistema})", fontweight='bold')
                plt.colorbar(im, ax=axes[idx], label='Valor normalizado')
            axes[idx].axis('off')

        # Oculta eixos vazios
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        if title is None:
            title = f"Análise Multiespectral - Amostra {amostra_id}"

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig

    def criar_grid_composicoes_falsas_cores(
        self,
        amostra_id: str,
        figsize: Tuple[int, int] = (18, 6),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Cria grid com diferentes composições RGB falsa-cor.

        Mostra:
        1. Composição SWIR (B06, B05, B02) - para argilas
        2. Composição VNIR (B03N, B02, B01) - composição natural
        3. Índice de Argilas em pseudo-cor

        Parameters
        ----------
        amostra_id : str
            ID da amostra
        figsize : Tuple
            Tamanho da figura
        title : str, optional
            Título customizado

        Returns
        -------
        plt.Figure
            Figura matplotlib com o grid
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. Composição SWIR (para alteração mineral)
        rgb_swir = self.carregador.criar_rgb_falsa_cor(
            amostra_id, banda_r='B06', banda_g='B05', banda_b='B02'
        )
        if rgb_swir is not None:
            axes[0].imshow(rgb_swir)
            axes[0].set_title('Composição SWIR\n(B06, B05, B02)\nDetecção de Argilas', fontweight='bold')
        axes[0].axis('off')

        # 2. Composição VNIR
        rgb_vnir = self.carregador.criar_rgb_falsa_cor(
            amostra_id, banda_r='B03N', banda_g='B02', banda_b='B01'
        )
        if rgb_vnir is not None:
            axes[1].imshow(rgb_vnir)
            axes[1].set_title('Composição VNIR\n(B03N, B02, B01)\nComposição Natural', fontweight='bold')
        axes[1].axis('off')

        # 3. Índice de Argilas
        indice = self.carregador.calcular_indice_mineral(
            amostra_id, banda_num='B06', banda_den1='B05', banda_den2='B04'
        )
        if indice is not None:
            im = axes[2].imshow(indice, cmap='hot')
            axes[2].set_title('Índice de Argilas\n(B06 / (B05+B04))', fontweight='bold')
            plt.colorbar(im, ax=axes[2], label='Intensidade')
        axes[2].axis('off')

        if title is None:
            title = f"Composições Espectrais - Amostra {amostra_id}"

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig

    def criar_grid_filtros_aplicados(
        self,
        amostra_id: str,
        banda: str = 'B06',
        figsize: Tuple[int, int] = (16, 4),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Cria grid mostrando a aplicação de diferentes filtros à mesma banda.

        Mostra:
        1. Original
        2. Gaussiano (sigma=1)
        3. Gaussiano (sigma=3)
        4. Realce de contraste

        Parameters
        ----------
        amostra_id : str
            ID da amostra
        banda : str
            Banda a filtrar
        figsize : Tuple
            Tamanho da figura
        title : str, optional
            Título customizado

        Returns
        -------
        plt.Figure
            Figura matplotlib com o grid
        """
        original = self.carregador.carregar_banda(amostra_id, banda, normalize=True)
        if original is None:
            logger.error(f"Não foi possível carregar banda {banda} da amostra {amostra_id}")
            return None

        fig, axes = plt.subplots(1, 4, figsize=figsize)

        # 1. Original
        im1 = axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original', fontweight='bold')
        plt.colorbar(im1, ax=axes[0])
        axes[0].axis('off')

        # 2. Gaussiano (sigma=1)
        filtrado_1 = self.carregador.aplicar_filtro_gaussiano(original, sigma=1.0)
        im2 = axes[1].imshow(filtrado_1, cmap='gray')
        axes[1].set_title('Gaussiano (σ=1)', fontweight='bold')
        plt.colorbar(im2, ax=axes[1])
        axes[1].axis('off')

        # 3. Gaussiano (sigma=3)
        filtrado_3 = self.carregador.aplicar_filtro_gaussiano(original, sigma=3.0)
        im3 = axes[2].imshow(filtrado_3, cmap='gray')
        axes[2].set_title('Gaussiano (σ=3)', fontweight='bold')
        plt.colorbar(im3, ax=axes[2])
        axes[2].axis('off')

        # 4. Realce de contraste (CLAHE-like)
        from scipy.ndimage import gaussian_filter

        blurred = gaussian_filter(original, sigma=0.5)
        realcado = original - blurred + 0.5  # Sharpening
        realcado = np.clip(realcado, 0, 1)
        im4 = axes[3].imshow(realcado, cmap='gray')
        axes[3].set_title('Realce (Unsharp Mask)', fontweight='bold')
        plt.colorbar(im4, ax=axes[3])
        axes[3].axis('off')

        if title is None:
            title = f"Aplicação de Filtros - Amostra {amostra_id}, Banda {banda}"

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig

    def criar_grid_comparativo_amostras(
        self,
        amostra_ids: List[str],
        bandas: List[str] = None,
        figsize: Tuple[int, int] = (20, 12),
        title: str = "Análise Comparativa de Amostras",
    ) -> plt.Figure:
        """
        Cria grid comparativo mostrando as mesmas bandas de diferentes amostras.

        Parameters
        ----------
        amostra_ids : List[str]
            IDs das amostras a comparar
        bandas : List[str], optional
            Bandas a visualizar. Se None, usa ['B01', 'B02', 'B06', 'B13']
        figsize : Tuple
            Tamanho da figura
        title : str
            Título do gráfico

        Returns
        -------
        plt.Figure
            Figura matplotlib com o grid
        """
        if bandas is None:
            bandas = ['B01', 'B02', 'B06', 'B13']

        n_amostras = len(amostra_ids)
        n_bandas = len(bandas)

        fig, axes = plt.subplots(n_bandas, n_amostras, figsize=figsize)
        if n_bandas == 1:
            axes = axes.reshape(1, -1)
        elif n_amostras == 1:
            axes = axes.reshape(-1, 1)

        # Itera sobre bandas (linhas) e amostras (colunas)
        for i, banda in enumerate(bandas):
            for j, amostra_id in enumerate(amostra_ids):
                ax = axes[i, j]
                imagem = self.carregador.carregar_banda(amostra_id, banda, normalize=True)

                if imagem is not None:
                    im = ax.imshow(imagem, cmap='gray')
                    if i == 0:  # Título no topo
                        ax.set_title(f"Amostra {amostra_id}", fontweight='bold')
                    if j == 0:  # Rótulo à esquerda
                        subsistema = self.carregador.BAND_INFO.get(banda, {}).get(
                            'subsistema', ''
                        )
                        ax.set_ylabel(f"{banda} ({subsistema})", fontweight='bold')
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"Banda {banda}\nnão encontrada",
                        ha='center',
                        va='center',
                        transform=ax.transAxes,
                    )

                ax.axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig

    def criar_grid_indices_minerais(
        self,
        amostra_ids: List[str],
        figsize: Tuple[int, int] = (18, 6),
        title: str = "Índices Minerais Detectados",
    ) -> plt.Figure:
        """
        Cria grid mostrando diferentes índices minerais para várias amostras.

        Calcula:
        1. Índice de Argilas (B06 / (B05 + B04))
        2. Índice de Carbonatos (B13 / B14)
        3. Índice de Óxidos de Fer (B02 / B01)

        Parameters
        ----------
        amostra_ids : List[str]
            IDs das amostras
        figsize : Tuple
            Tamanho da figura
        title : str
            Título do gráfico

        Returns
        -------
        plt.Figure
            Figura matplotlib com o grid
        """
        n_amostras = len(amostra_ids)
        fig, axes = plt.subplots(3, n_amostras, figsize=figsize)

        if n_amostras == 1:
            axes = axes.reshape(3, 1)

        # Linha 1: Índice de Argilas
        for j, amostra_id in enumerate(amostra_ids):
            indice = self.carregador.calcular_indice_mineral(
                amostra_id, banda_num='B06', banda_den1='B05', banda_den2='B04'
            )
            if indice is not None:
                im = axes[0, j].imshow(indice, cmap='hot')
                axes[0, j].set_title(f"Amostra {amostra_id}", fontweight='bold')
                if j == n_amostras - 1:
                    plt.colorbar(im, ax=axes[0, j], label='Intensidade')
            axes[0, j].axis('off')

        # Linha 2: Índice de Carbonatos
        for j, amostra_id in enumerate(amostra_ids):
            indice = self.carregador.calcular_indice_mineral(
                amostra_id, banda_num='B13', banda_den1='B14'
            )
            if indice is not None:
                im = axes[1, j].imshow(indice, cmap='cool')
                if j == n_amostras - 1:
                    plt.colorbar(im, ax=axes[1, j], label='Intensidade')
            axes[1, j].axis('off')

        # Linha 3: Índice de Óxidos de Ferro
        for j, amostra_id in enumerate(amostra_ids):
            indice = self.carregador.calcular_indice_mineral(
                amostra_id, banda_num='B02', banda_den1='B01'
            )
            if indice is not None:
                im = axes[2, j].imshow(indice, cmap='copper')
                if j == n_amostras - 1:
                    plt.colorbar(im, ax=axes[2, j], label='Intensidade')
            axes[2, j].axis('off')

        # Rótulos das linhas
        axes[0, 0].set_ylabel('Índice Argilas\n(B06/(B05+B04))', fontweight='bold')
        axes[1, 0].set_ylabel('Índice Carbonatos\n(B13/B14)', fontweight='bold')
        axes[2, 0].set_ylabel('Índice Fe-Óxidos\n(B02/B01)', fontweight='bold')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig
