def prepare_dataset(dataset_path: str, extracted_codes_path: str):
    """
    Prepares the dataset by loading the pixel file, extracting image ID,
    creating the 'label' column, and returning features (X) and labels (y).

    :param dataset_path: Path to the pixel dataset CSV file.
    :param extracted_codes_path: Path to the JSON file containing positive and negative codes.
    :return: X (DataFrame of features), y (Series of labels)
    """
    import json
    import re
    import pandas as pd

    df = pd.read_csv(dataset_path)

    # Load extracted codes
    with open(extracted_codes_path, 'r') as f:
        extracted_codes = json.load(f)

    # Extract image ID (re-using the logic from previous steps)
    all_ids = extracted_codes['negativos'] + extracted_codes['positivos']
    all_ids_sorted = sorted(all_ids, key=len, reverse=True)
    id_pattern = '|'.join(re.escape(id_val) for id_val in all_ids_sorted)
    regex = rf'({id_pattern})'
    image_id = df['path'].str.extract(regex)[0]

    # Create label column
    label = image_id.apply(
        lambda x: 1 if x in extracted_codes['positivos'] else (
            0 if x in extracted_codes['negativos'] else -1
        )
    )

    df = pd.concat([df, image_id.rename('image_id'), label.rename('label')], axis=1)

    # Drop non-feature columns and separate X and y
    X = df.drop(
        columns=[
            'path', 'filename', 'count', 'height', 'width', 'dtype',
            'crs', 'transform', 'image_id', 'label'
        ]
    )
    y = df["label"]

    return X, y


def prepare_dataset_with_groups(
    dataset_path: str,
    extracted_codes_path: str,
    drop_invalid: bool = True,
):
    """
    Prepara o dataset para classificação e retorna também os grupos por imagem.

    Parameters
    ----------
    dataset_path : str
        Caminho do CSV com pixels.
    extracted_codes_path : str
        Caminho do JSON com listas de códigos positivos e negativos.
    drop_invalid : bool, default=True
        Se True, remove rótulos inválidos (-1).

    Returns
    -------
    tuple
        (X, y, groups, df_prepared, feature_cols)
    """
    import json
    import re
    import numpy as np
    import pandas as pd

    df = pd.read_csv(dataset_path)

    with open(extracted_codes_path, "r", encoding="utf-8") as f:
        extracted_codes = json.load(f)

    positivos = set(extracted_codes.get("positivos", []))
    negativos = set(extracted_codes.get("negativos", []))

    all_ids = sorted(list(positivos | negativos), key=len, reverse=True)
    if not all_ids:
        raise ValueError("Lista de codigos vazia em extracted_codes.json")

    id_pattern = "|".join(re.escape(id_val) for id_val in all_ids)
    image_id = df["path"].astype(str).str.extract(rf"({id_pattern})", expand=False)

    label = image_id.apply(
        lambda x: 1 if x in positivos else (0 if x in negativos else -1)
    )

    df = pd.concat(
        [df, image_id.rename("image_id"), label.rename("label")], axis=1
    )

    if drop_invalid:
        df = df[df["label"] != -1].copy()

    df = df.dropna(subset=["image_id"]).copy()
    df["label"] = df["label"].replace({np.nan: -1}).astype(int)

    feature_cols = [c for c in df.columns if c.startswith("pixel_")]
    if not feature_cols:
        drop_cols = {
            "path",
            "filename",
            "count",
            "height",
            "width",
            "dtype",
            "crs",
            "transform",
            "image_id",
            "label",
        }
        feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df["label"].copy()
    groups = df["image_id"].copy()

    return X, y, groups, df, feature_cols
