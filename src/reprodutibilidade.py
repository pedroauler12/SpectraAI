"""
Controles globais de reprodutibilidade para experimentos de ML.

Para usar nos demais arquivos, coloque no início:
from src.utils.reprodutibilidade import set_global_seed

set_global_seed(42)

e 

random_state=42 precisa estar em todos os modelos

Isso garante que duas reproduções do mesmo código deem o mesmo resultado, diminui a aleatoriedade.


"""

import os
import random
import numpy as np


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Define seeds globais e controles de determinismo.

    Parameters
    ----------
    seed : int
        Seed principal do experimento.
    deterministic : bool
        Se True, reduz variações por paralelismo.
    """

    random.seed(seed)

    np.random.seed(seed)

    # Hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Controle de threads
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    print(f"[Reprodutibilidade] Seed definida: {seed}")
