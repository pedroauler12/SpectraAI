def min_max_normalize_bands(data_list):
    normalized_data_list = []
    for matrix in data_list:
        # Assuming matrix shape is (bands, height, width)
        num_bands = matrix.shape[0]
        normalized_matrix = np.zeros_like(matrix, dtype=np.float32)
        for band_idx in range(num_bands):
            band_data = matrix[band_idx]
            min_val = np.min(band_data)
            max_val = np.max(band_data)

            if max_val == min_val:
                # Avoid division by zero if all values in the band are the same
                normalized_matrix[band_idx] = 0.0
            else:
                normalized_matrix[band_idx] = (band_data - min_val) / (max_val - min_val)
        normalized_data_list.append(normalized_matrix)
    return normalized_data_list

# Apply the normalization function
normalized_matrices = min_max_normalize_bands(data_matrices)

print(f"Normalização Min-Max aplicada a {len(normalized_matrices)} matrizes.")