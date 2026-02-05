# -*- coding: utf-8 -*-
"""
Módulo para cálculo de estatísticas básicas por banda de imagens de satélite ASTER.

As estatísticas são calculadas sobre os VALORES DOS PIXELS de cada banda espectral.
Cada pixel contém um valor de reflectância ou radiância dependendo do subsistema.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import rasterio
import pandas as pd
from dataclasses import dataclass, asdict
import json
import re


@dataclass
class BandStatistics:
    """
    Estatísticas calculadas sobre os VALORES DOS PIXELS de uma banda espectral.
    
    Os valores estatísticos (média, mediana, etc.) representam as medidas dos 
    valores de reflectância/radiância dos pixels da imagem.
    """
    subsistema: str  # VNIR, SWIR ou TIR
    banda: str  # B01, B02, B03N, B04-B09, B10-B14
    arquivo: str
    media: float  # Média dos valores dos pixels
    mediana: float  # Mediana dos valores dos pixels
    desvio_padrao: float  # Desvio padrão dos valores dos pixels
    variancia: float  # Variância dos valores dos pixels
    minimo: float  # Valor mínimo de pixel
    maximo: float  # Valor máximo de pixel
    percentil_25: float
    percentil_50: float
    percentil_75: float
    percentil_90: float
    percentil_95: float
    percentil_99: float
    iqr: float  # Intervalo interquartil
    mad: float  # Median Absolute Deviation (estatística robusta)
    cv: float  # Coeficiente de variação (%)
    pixels_validos: int  # Quantidade de pixels válidos usados
    pixels_totais: int  # Quantidade total de pixels na imagem
    
    def to_dict(self):
        return asdict(self)


def extrair_subsistema_banda(nome_arquivo: str) -> tuple:
    """
    Extrai o subsistema e número da banda do nome do arquivo ASTER.
    
    Exemplos:
        'AST_L1T_..._VNIR_B01.tif' -> ('VNIR', 'B01')
        'AST_L1T_..._TIR_B10.tif' -> ('TIR', 'B10')
        'AST_L1T_..._SWIR_B04.tif' -> ('SWIR', 'B04')
    
    Returns:
        tuple: (subsistema, banda) ou (None, None) se não encontrar
    """
    # Padrão: (VNIR|SWIR|TIR)_B\d+
    match = re.search(r'(VNIR|SWIR|TIR)_(B\d+[A-Z]*)', nome_arquivo)
    if match:
        return match.group(1), match.group(2)
    return None, None


def calcular_estatisticas_banda(imagem_path, percentis=[25,50,75,90,95,99], ignorar_nodata=True):
    """
    Calcula estatísticas sobre os VALORES DOS PIXELS de uma banda espectral.
    
    Args:
        imagem_path: Caminho do arquivo GeoTIFF
        percentis: Lista de percentis a calcular
        ignorar_nodata: Se True, ignora pixels NoData (geralmente valor 0)
        
    Returns:
        BandStatistics com todas as métricas calculadas dos valores dos pixels
        
    Raises:
        ValueError: Se não houver pixels válidos
        rasterio.errors.RasterioIOError: Se houver erro na leitura do arquivo
    """
    imagem_path = Path(imagem_path)
    
    # Extrai subsistema e banda do nome do arquivo
    subsistema, banda = extrair_subsistema_banda(imagem_path.name)
    
    if not subsistema or not banda:
        # Fallback: tenta extrair do nome completo
        subsistema = "DESCONHECIDO"
        banda = imagem_path.stem
    
    try:
        with rasterio.open(imagem_path) as src:
            # Lê os valores dos pixels da banda
            banda_data = src.read(1)
            nodata = src.nodata
            
            # Filtra pixels válidos
            if ignorar_nodata and nodata is not None:
                mascara_valida = banda_data != nodata
                dados_validos = banda_data[mascara_valida]
            else:
                dados_validos = banda_data.flatten()
            
            # Remove infinitos e NaN
            dados_validos = dados_validos[np.isfinite(dados_validos)]
            
            pixels_validos = len(dados_validos)
            pixels_totais = banda_data.size
            
            if pixels_validos == 0:
                raise ValueError(f"Nenhum pixel válido em {imagem_path.name}")
            
            # Calcula estatísticas dos valores dos pixels
            media = float(np.mean(dados_validos))
            mediana = float(np.median(dados_validos))
            desvio_padrao = float(np.std(dados_validos))
            variancia = float(np.var(dados_validos))
            minimo = float(np.min(dados_validos))
            maximo = float(np.max(dados_validos))
            
            # Calcula percentis
            p25, p50, p75, p90, p95, p99 = np.percentile(dados_validos, percentis)
            
            # Estatísticas robustas
            iqr = float(p75 - p25)
            mad = float(np.median(np.abs(dados_validos - mediana)))
            cv = float((desvio_padrao / media * 100) if media != 0 else 0)
            
            return BandStatistics(
                subsistema=subsistema,
                banda=banda,
                arquivo=imagem_path.name,
                media=media,
                mediana=mediana,
                desvio_padrao=desvio_padrao,
                variancia=variancia,
                minimo=minimo,
                maximo=maximo,
                percentil_25=p25,
                percentil_50=p50,
                percentil_75=p75,
                percentil_90=p90,
                percentil_95=p95,
                percentil_99=p99,
                iqr=iqr,
                mad=mad,
                cv=cv,
                pixels_validos=pixels_validos,
                pixels_totais=pixels_totais
            )
    except rasterio.errors.RasterioIOError as e:
        raise IOError(f"Erro ao ler {imagem_path.name}: {str(e)}")


def processar_pasta_imagens(pasta_path):
    """
    Processa todas as bandas individuais em uma pasta.
    
    Filtra apenas arquivos de banda individual (com _B no nome),
    ignorando arquivos compostos (VNIR.tif, TIR.tif, SWIR.tif).
    """
    pasta_path = Path(pasta_path)
    if not pasta_path.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {pasta_path}")
    
    # Filtra apenas bandas individuais
    arquivos = [f for f in pasta_path.glob("*.tif") if '_B' in f.stem]
    
    if not arquivos:
        print(f"  Nenhuma banda encontrada")
        return []
    
    print(f"  Processando {len(arquivos)} bandas...")
    
    resultados = []
    erros = []
    
    for i, arq in enumerate(sorted(arquivos), 1):
        try:
            stats = calcular_estatisticas_banda(arq)
            resultados.append(stats)
            print(f"     [{i}/{len(arquivos)}] {stats.subsistema}/{stats.banda}")
        except IOError as e:
            erros.append((arq.name, str(e)))
            print(f"     [{i}/{len(arquivos)}] {arq.name} - Erro de leitura")
        except ValueError as e:
            erros.append((arq.name, str(e)))
            print(f"     [{i}/{len(arquivos)}] {arq.name} - Sem pixels válidos")
        except Exception as e:
            erros.append((arq.name, str(e)))
            print(f"     [{i}/{len(arquivos)}] {arq.name} - {type(e).__name__}")
    
    if erros:
        print(f"\n  {len(erros)} erro(s) encontrado(s)")
        
    return resultados


def processar_multiplas_pastas(pasta_base):
    """Processa múltiplas pastas de cenas ASTER."""
    pasta_base = Path(pasta_base)
    subpastas = [p for p in pasta_base.iterdir() if p.is_dir()]
    
    print(f"Encontradas {len(subpastas)} cenas ASTER\n")
    
    resultados = {}
    for idx, pasta in enumerate(sorted(subpastas), 1):
        print(f"[{idx}/{len(subpastas)}] 📁 {pasta.name}")
        stats = processar_pasta_imagens(pasta)
        if stats:
            resultados[pasta.name] = stats
            print(f"  {len(stats)} bandas processadas\n")
        else:
            print(f"     Nenhuma banda processada\n")
    
    return resultados


def estatisticas_para_dataframe(estatisticas):
    """Converte estatísticas para DataFrame com colunas organizadas."""
    dados = []
    for pasta, stats_list in estatisticas.items():
        for stats in stats_list:
            d = stats.to_dict()
            d['cena'] = pasta
            dados.append(d)
    
    df = pd.DataFrame(dados)
    
    # Reorganiza colunas para melhor visualização
    colunas_ordenadas = [
        'cena', 'subsistema', 'banda', 'arquivo',
        'media', 'mediana', 'desvio_padrao', 'minimo', 'maximo',
        'percentil_25', 'percentil_50', 'percentil_75', 'percentil_90', 'percentil_95', 'percentil_99',
        'iqr', 'mad', 'cv', 'variancia',
        'pixels_validos', 'pixels_totais'
    ]
    
    return df[colunas_ordenadas]


def gerar_estatisticas_agregadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera estatísticas COMPARANDO as diferentes bandas espectrais.
    
    Agrupa por subsistema e banda para calcular:
    - Quantas vezes cada banda aparece
    - Média/mediana dos valores ENTRE diferentes imagens da mesma banda
    - Qual banda tem maiores/menores valores
    
    Args:
        df: DataFrame com estatísticas individuais por banda
        
    Returns:
        DataFrame com estatísticas agregadas por banda
    """
    print("\n Gerando estatísticas agregadas por banda...")
    
    # Agrupa por subsistema e banda
    agregado = df.groupby(['subsistema', 'banda']).agg({
        'media': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'mediana': ['mean', 'median'],
        'desvio_padrao': ['mean', 'median'],
        'minimo': ['min', 'mean'],
        'maximo': ['max', 'mean'],
        'cv': ['mean', 'median'],
        'pixels_validos': ['mean', 'sum']
    }).round(2)
    
    # Achata os nomes das colunas
    agregado.columns = ['_'.join(col).strip() for col in agregado.columns.values]
    agregado = agregado.reset_index()
    
    # Renomeia para ficar mais claro
    agregado = agregado.rename(columns={
        'media_count': 'ocorrencias',
        'media_mean': 'media_entre_cenas',
        'media_median': 'mediana_entre_cenas',
        'media_std': 'desvio_entre_cenas',
        'media_min': 'menor_media',
        'media_max': 'maior_media',
        'pixels_validos_mean': 'media_pixels_validos',
        'pixels_validos_sum': 'total_pixels_analisados'
    })
    
    return agregado


def gerar_resumo_por_subsistema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera resumo estatístico por subsistema (VNIR, SWIR, TIR).
    
    Args:
        df: DataFrame com estatísticas individuais
        
    Returns:
        DataFrame com resumo por subsistema
    """
    print("\n Gerando resumo por subsistema...")
    
    resumo = df.groupby('subsistema').agg({
        'banda': 'count',
        'media': ['mean', 'median', 'std'],
        'desvio_padrao': 'mean',
        'cv': 'mean',
        'pixels_validos': 'sum'
    }).round(2)
    
    resumo.columns = ['_'.join(col).strip() for col in resumo.columns.values]
    resumo = resumo.reset_index()
    
    resumo = resumo.rename(columns={
        'banda_count': 'quantidade_bandas',
        'media_mean': 'media_geral',
        'media_median': 'mediana_geral',
        'media_std': 'variacao_entre_bandas',
        'desvio_padrao_mean': 'desvio_medio',
        'cv_mean': 'cv_medio',
        'pixels_validos_sum': 'total_pixels'
    })
    
    return resumo


def salvar_estatisticas(estatisticas, output_path, formato='csv'):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if formato == 'csv':
        df = estatisticas_para_dataframe(estatisticas)
        df.to_csv(output_path, index=False)
        print(f"Salvo: {output_path}")
    elif formato == 'json':
        dados = {p: [s.to_dict() for s in sl] for p, sl in estatisticas.items()}
        with open(output_path, 'w') as f:
            json.dump(dados, f, indent=2)
        print(f"Salvo: {output_path}")


if __name__ == "__main__":
    PASTA_ASTER = Path(__file__).resolve().parent.parent.parent / "dados_dropbox" / "ASTER"
    OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
    
    print("Iniciando análise de bandas ASTER\n")
    print("="*80)
    
    try:
        # Processa todas as cenas
        stats = processar_multiplas_pastas(PASTA_ASTER)
        
        if not stats:
            print("\n Nenhuma banda processada!")
            exit(1)
        
        # Converte para DataFrame
        df_individual = estatisticas_para_dataframe(stats)
        
        print("="*80)
        print(f"\n Total: {len(df_individual)} bandas processadas\n")
        
        # Salva estatísticas individuais
        print("Salvando estatísticas individuais...")
        salvar_estatisticas(stats, OUTPUT_DIR / "estatisticas_bandas_individual.json", 'json')
        
        # Gera e salva estatísticas agregadas por banda
        df_agregado = gerar_estatisticas_agregadas(df_individual)
        output_agregado = OUTPUT_DIR / "estatisticas_bandas_agregadas.json"
        df_agregado.to_json(output_agregado, orient='records', indent=2)
        print(f"Salvo: {output_agregado}")
        
        # Gera e salva resumo por subsistema
        df_subsistema = gerar_resumo_por_subsistema(df_individual)
        output_subsistema = OUTPUT_DIR / "estatisticas_por_subsistema.json"
        df_subsistema.to_json(output_subsistema, orient='records', indent=2)
        print(f"Salvo: {output_subsistema}")
        
        # Exibe resumos
        print("\n" + "="*80)
        print("RESUMO POR SUBSISTEMA")
        print("="*80)
        print(df_subsistema.to_string(index=False))
        
        print("\n" + "="*80)
        print("TOP BANDAS POR MÉDIA GERAL")
        print("="*80)
        top_bandas = df_agregado.nlargest(10, 'media_entre_cenas')[
            ['subsistema', 'banda', 'ocorrencias', 'media_entre_cenas', 'mediana_entre_cenas']
        ]
        print(top_bandas.to_string(index=False))
        
        print("\n" + "="*80)
        print("DISTRIBUIÇÃO DE BANDAS POR SUBSISTEMA")
        print("="*80)
        distribuicao = df_individual.groupby('subsistema')['banda'].value_counts().reset_index()
        distribuicao.columns = ['subsistema', 'banda', 'contagem']
        print(distribuicao.head(15).to_string(index=False))
        
        print("\n" + "="*80)
        print("Análise concluída com sucesso!")
        print("="*80)
        
    except Exception as e:
        print(f"\n Erro: {e}")
        import traceback
        traceback.print_exc()
