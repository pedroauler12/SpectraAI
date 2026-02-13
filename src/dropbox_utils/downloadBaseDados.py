"""
Módulo para download e sincronização de arquivos e pastas de links públicos do Dropbox.
"""

import os
import requests
import zipfile
from pathlib import Path
from typing import Optional, Set, Tuple
import shutil
import hashlib
from datetime import datetime


def download_dropbox_folder(
    dropbox_url: str,
    output_folder: str = "dados_dropbox",
    force_download: bool = False,
    extract_zip: bool = True,
    sync_mode: bool = True
) -> Path:
    """
    Faz download e sincronização de arquivos ou pastas de um link público do Dropbox.
    
    Args:
        dropbox_url: URL pública do Dropbox (link compartilhado)
        output_folder: Nome da pasta de destino na raiz do projeto (padrão: 'dados_dropbox')
        force_download: Se True, redownload completo ignorando sincronização (padrão: False)
        extract_zip: Se True, extrai automaticamente arquivos ZIP (padrão: True)
        sync_mode: Se True, sincroniza mantendo apenas arquivos atuais (padrão: True)
    
    Returns:
        Path: Caminho para a pasta de destino
        
    Exemplo:
        >>> from src.dropbox_utils.downloadBaseDados import download_dropbox_folder
        >>> download_dropbox_folder('https://www.dropbox.com/...')
    """
    
    # Encontra a raiz do projeto (assume que está 2 níveis acima deste arquivo)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    
    # Cria o caminho de destino
    destination = project_root / output_folder
    temp_sync_folder = project_root / f"{output_folder}_temp_sync"
    
    # Converte URL do Dropbox para download direto
    download_url = _convert_dropbox_url(dropbox_url)
    
    # Nome do arquivo temporário
    temp_file = temp_sync_folder / "temp_download.zip"
    temp_sync_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Pasta de destino: {destination}")
        print(f"Baixando do Dropbox...")
        _download_file(download_url, temp_file)
        
        # Extrai para pasta temporária
        if temp_file.suffix == '.zip' and extract_zip:
            print(f"Extraindo arquivo ZIP...")
            _extract_zip(temp_file, temp_sync_folder)
            temp_file.unlink()
        
        # Se sync_mode está ativado e a pasta de destino existe
        if sync_mode and destination.exists():
            print(f"Sincronizando arquivos...")
            _sync_folders(temp_sync_folder, destination)
        else:
            # Modo download completo
            print(f"Copiando arquivos...")
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=True)
            _copy_folder_contents(temp_sync_folder, destination)
        
        # Remove pasta temporária
        shutil.rmtree(temp_sync_folder)
        
        print(f"Sincronização concluída!")
            
    except Exception as e:
        print(f"Erro durante o download: {e}")
        # Limpa pasta temporária em caso de erro
        if temp_sync_folder.exists():
            shutil.rmtree(temp_sync_folder)
        raise
    
    return destination


def _convert_dropbox_url(url: str) -> str:
    """
    Converte URL de compartilhamento do Dropbox em URL de download direto.
    
    Args:
        url: URL pública do Dropbox
        
    Returns:
        str: URL modificada para download direto
    """
    # Para URLs do tipo /scl/ (novos links compartilhados), apenas troca dl=0 por dl=1
    if '/scl/' in url:
        # Substitui dl=0 por dl=1 ou adiciona dl=1 se não existir
        if 'dl=0' in url:
            return url.replace('dl=0', 'dl=1')
        elif '?' in url:
            return url + '&dl=1'
        else:
            return url + '?dl=1'
    
    # Para URLs antigas (sem /scl/), usa o método de substituição de domínio
    # Remove parâmetros da URL
    base_url = url.split('?')[0]
    
    # Substitui o domínio
    if 'www.dropbox.com' in base_url:
        return base_url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
    elif 'dropbox.com' in base_url:
        return base_url.replace('dropbox.com', 'dl.dropboxusercontent.com')
    
    # Fallback: adiciona dl=1
    return url.rstrip('/') + ('&dl=1' if '?' in url else '?dl=1')


def _download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """
    Faz download de um arquivo com barra de progresso.
    
    Args:
        url: URL do arquivo
        destination: Caminho de destino
        chunk_size: Tamanho do chunk para download (bytes)
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        if total_size == 0:
            # Sem informação de tamanho
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Mostra progresso
                    percent = (downloaded / total_size) * 100
                    print(f"\r   Progresso: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end='')
            print()  # Nova linha após completar


def _extract_zip(zip_path: Path, destination: Path) -> None:
    """
    Extrai um arquivo ZIP.
    
    Args:
        zip_path: Caminho do arquivo ZIP
        destination: Pasta de destino
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination)


def _get_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Calcula o hash MD5 de um arquivo.
    
    Args:
        file_path: Caminho do arquivo
        chunk_size: Tamanho do chunk para leitura
        
    Returns:
        str: Hash MD5 do arquivo
    """
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def _get_all_files(folder: Path, relative_to: Path = None) -> Set[Path]:
    """
    Retorna todos os arquivos em uma pasta recursivamente.
    
    Args:
        folder: Pasta para listar arquivos
        relative_to: Pasta base para caminhos relativos
        
    Returns:
        Set[Path]: Conjunto de caminhos relativos dos arquivos
    """
    if relative_to is None:
        relative_to = folder
    
    files = set()
    for item in folder.rglob('*'):
        if item.is_file():
            files.add(item.relative_to(relative_to))
    return files


def _copy_folder_contents(source: Path, destination: Path) -> None:
    """
    Copia conteúdo de uma pasta para outra, ignorando pastas temporárias.
    
    Args:
        source: Pasta de origem
        destination: Pasta de destino
    """
    for item in source.iterdir():
        # Ignora arquivos temporários e arquivos ZIP
        if item.name.startswith('temp_') or item.suffix == '.zip':
            continue
            
        dest_item = destination / item.name
        if item.is_dir():
            shutil.copytree(item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_item)


def _sync_folders(source: Path, destination: Path) -> None:
    """
    Sincroniza duas pastas: adiciona novos arquivos, atualiza modificados e remove deletados.
    
    Args:
        source: Pasta de origem (novos dados do Dropbox)
        destination: Pasta de destino (dados locais)
    """
    # Obtém listas de arquivos
    source_files = _get_all_files(source)
    dest_files = _get_all_files(destination)
    
    # Arquivos novos ou atualizados
    added = 0
    updated = 0
    removed = 0
    unchanged = 0
    
    # Processa arquivos da origem
    for rel_path in source_files:
        source_file = source / rel_path
        dest_file = destination / rel_path
        
        # Cria diretórios se necessário
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not dest_file.exists():
            # Arquivo novo
            shutil.copy2(source_file, dest_file)
            added += 1
        else:
            # Verifica se o arquivo foi modificado comparando hash
            if _get_file_hash(source_file) != _get_file_hash(dest_file):
                shutil.copy2(source_file, dest_file)
                updated += 1
            else:
                unchanged += 1
    
    # Remove arquivos que não existem mais na origem
    for rel_path in dest_files - source_files:
        dest_file = destination / rel_path
        if dest_file.exists():
            dest_file.unlink()
            removed += 1
    
    # Limpa pastas vazias
    _remove_empty_folders(destination)

def _remove_empty_folders(folder: Path) -> None:
    """
    Remove pastas vazias recursivamente.
    
    Args:
        folder: Pasta para limpar
    """
    for item in sorted(folder.rglob('*'), reverse=True):
        if item.is_dir() and not any(item.iterdir()):
            item.rmdir()


def limpar_dados(output_folder: str = "dados_dropbox") -> None:
    """
    Remove a pasta de dados baixados do Dropbox.
    
    Args:
        output_folder: Nome da pasta a ser removida (padrão: 'dados_dropbox')
        
    Exemplo:
        >>> from src.dropbox_utils.downloadBaseDados import limpar_dados
        >>> limpar_dados()
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    destination = project_root / output_folder
    
    if destination.exists():
        shutil.rmtree(destination)
        print(f"Pasta {destination} removida com sucesso!")
    else:
        print(f"Pasta {destination} não existe.")


if __name__ == "__main__":
    # Quando executado diretamente, faz o download
    DROPBOX_URL = "https://www.dropbox.com/scl/fo/5w7xvlpanxdeu5lgt4a49/AMFZYG1dcnnUqCr8uCQLQ64/Coordenadas%20Minas%20com%20Terras%20Raras?rlkey=aolxn05zq4aqw09wz68297wa0&st=gg242w8w&dl=0"
    try:
        resultado = download_dropbox_folder(DROPBOX_URL, sync_mode=True)
        print()
        print(f"Sucesso! Dados sincronizados em: {resultado}")
    except Exception as e:
        print()
        print(f"Erro: {e}")