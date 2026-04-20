from pathlib import Path
import numpy as np
from scipy.io import loadmat


def inspect_mat_file(file_path):
    """
    Print top-level keys and basic info.
    """
    data = loadmat(file_path)

    print(f"\n{'=' * 60}")
    print(f"FILE: {file_path}")
    print(f"{'=' * 60}")

    for key, value in data.items():
        if key.startswith("__"):
            continue

        print(f"Key: {key}")
        print(f"Type: {type(value)}")

        if isinstance(value, np.ndarray):
            print(f"Shape: {value.shape}")
            print(f"Dtype: {value.dtype}")

            # Only print small preview
            try:
                preview = value.flatten()[:3]
                print(f"Preview: {preview}")
            except Exception:
                pass

        else:
            print(f"Value: {value}")

        print("-" * 40)


def collect_numeric_arrays(obj, found=None, path="root"):
    """
    Recursively collect numeric numpy arrays from nested MATLAB-loaded objects.
    """
    if found is None:
        found = []

    if isinstance(obj, np.ndarray):
        # Case 1: direct numeric array
        if np.issubdtype(obj.dtype, np.number):
            found.append((path, obj))
        # Case 2: object array -> recurse into elements
        elif obj.dtype == object:
            for idx, item in np.ndenumerate(obj):
                collect_numeric_arrays(item, found, path=f"{path}{idx}")

    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            collect_numeric_arrays(item, found, path=f"{path}[{i}]")

    return found


def extract_all_candidate_signals(file_path):
    """
    Return all numeric arrays found inside the .mat file.
    """
    data = loadmat(file_path)
    candidates = []

    for key, value in data.items():
        if key.startswith("__"):
            continue

        found = collect_numeric_arrays(value, path=key)
        candidates.extend(found)

    return candidates


def choose_best_signal_array(candidates, min_length=256):
    """
    Choose the most likely vibration array from all numeric arrays found.
    Preference:
    - 2D arrays where one dimension is reasonably large
    - otherwise large 1D arrays
    """
    filtered = []

    for path, arr in candidates:
        arr = np.asarray(arr)

        # Skip scalars or tiny arrays
        if arr.size < min_length:
            continue

        filtered.append((path, arr))

    if not filtered:
        return None, None

    # Prefer 2D arrays like (10, 12000)
    two_d = [(p, a) for p, a in filtered if a.ndim == 2]
    if two_d:
        two_d.sort(key=lambda x: x[1].size, reverse=True)
        return two_d[0]

    # Then prefer 1D arrays
    one_d = [(p, a) for p, a in filtered if a.ndim == 1]
    if one_d:
        one_d.sort(key=lambda x: x[1].size, reverse=True)
        return one_d[0]

    # Fallback: largest numeric array
    filtered.sort(key=lambda x: x[1].size, reverse=True)
    return filtered[0]


def extract_signal_matrix(file_path):
    """
    Extract the main signal matrix from a nested .mat file.

    Returns
    -------
    signal_key : str
        Path/key of the selected array
    signal_matrix : np.ndarray
        2D array of shape (n_segments, signal_length)
    """
    candidates = extract_all_candidate_signals(file_path)
    signal_key, signal = choose_best_signal_array(candidates)

    if signal is None:
        raise ValueError(f"Could not find numeric signal in file: {file_path}")

    signal = np.asarray(signal)

    # Convert 1D to 2D
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)

    # If shape is (signal_length, n_segments), transpose if needed
    # We usually prefer rows = samples/segments, cols = signal length
    if signal.ndim == 2 and signal.shape[0] > signal.shape[1]:
        # Example: (12000, 10) should become (10, 12000)
        signal = signal.T

    return signal_key, signal


def flatten_signal_matrix(signal_matrix):
    """
    Ensure output is a clean 2D numeric array.
    """
    signal_matrix = np.asarray(signal_matrix, dtype=np.float64)

    if signal_matrix.ndim != 2:
        raise ValueError(f"Expected 2D signal matrix, got shape {signal_matrix.shape}")

    return signal_matrix


if __name__ == "__main__":
    data_dir = Path(r"F:\NeuTech\CWRU")

    for file_path in data_dir.glob("*.mat"):
        inspect_mat_file(file_path)

    print("\n" + "=" * 60)
    print("Testing signal extraction...")
    print("=" * 60)

    for file_path in data_dir.glob("*.mat"):
        key, signal_matrix = extract_signal_matrix(file_path)
        signal_matrix = flatten_signal_matrix(signal_matrix)

        print(f"{file_path.name}")
        print(f"  selected key/path : {key}")
        print(f"  extracted shape   : {signal_matrix.shape}")
        print(f"  dtype             : {signal_matrix.dtype}")
        print(f"  first 5 values    : {signal_matrix[0, :5]}")
        print("-" * 40)