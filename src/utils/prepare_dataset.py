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
