# src/data/load_data.py
import pandas as pd
import h5py
import os


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load AIS data from HDF5, CSV or other format.
    Currently assumes HDF5 structure from previous preprocessing.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
        print(f"Loading HDF5 file: {data_path}")
        with h5py.File(data_path, 'r') as f:
            # Example structure – adjust to your actual keys
            if 'df' in f:
                df = pd.DataFrame(f['df'][:])
                df.columns = f['columns'][:].astype(str)
            else:
                # Assume you stored multiple trajectories
                trajectories = []
                for key in f.keys():
                    if key.startswith('traj_'):
                        traj = pd.DataFrame(f[key][:])
                        traj['mmsi'] = key.split('_')[1]  # example
                        trajectories.append(traj)
                df = pd.concat(trajectories, ignore_index=True)
    else:
        # fallback: CSV
        df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} rows")
    return df
