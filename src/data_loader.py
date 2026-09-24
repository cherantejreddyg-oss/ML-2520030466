import os
import pandas as pd

DEFAULT_DATA_PATH = os.path.join("data", "AccidentProneAreas_Complete(Complete Dataset) (3) (1).csv")

def load_raw_dataset(file_path=None):
    """
    Loads raw CSV dataset, strips metadata title header rows,
    standardizes initial column names, and handles basic missing file errors.
    """
    if file_path is None:
        file_path = DEFAULT_DATA_PATH
    
    if not os.path.exists(file_path):
        # Fallback search if path differs slightly
        possible_dir = "data"
        if os.path.exists(possible_dir):
            csv_files = [f for f in os.listdir(possible_dir) if f.endswith(".csv")]
            if csv_files:
                file_path = os.path.join(possible_dir, csv_files[0])
            else:
                raise FileNotFoundError(f"No CSV file found in '{possible_dir}' directory.")
        else:
            raise FileNotFoundError(f"Dataset file not found at '{file_path}'.")

    try:
        # The CSV contains 4 header/banner rows before data:
        # Row 0: Title Banner
        # Row 1: District Name Banner
        # Row 2: First-level headers
        # Row 3: Sub-level headers
        df_raw = pd.read_csv(file_path, skiprows=4, header=None)
        
        standard_columns = [
            'sl_no', 'zone', 'police_station', 'road_info', 'location_name',
            'start_km', 'end_km', 'accidents_2023', 'accidents_2024', 'accidents_2025',
            'total_accidents', 'fatalities_2023', 'fatalities_2024', 'fatalities_2025', 'total_fatalities'
        ]
        
        # If columns exceed expected, assign standard names up to available columns
        if len(df_raw.columns) >= len(standard_columns):
            df_raw = df_raw.iloc[:, :len(standard_columns)]
            df_raw.columns = standard_columns
        else:
            df_raw.columns = standard_columns[:len(df_raw.columns)]

        # Drop completely blank rows if any
        df_raw = df_raw.dropna(how='all').reset_index(drop=True)

        return df_raw, file_path

    except Exception as e:
        raise RuntimeError(f"Error reading dataset CSV: {str(e)}")

def display_load_info(df, path):
    """
    Displays clean dataset load status.
    """
    print("\n" + "="*60)
    print("           DATASET LOAD STATUS")
    print("="*60)
    print(f"Status             : Dataset loaded successfully!")
    print(f"File Path          : {path}")
    print(f"Number of Records  : {df.shape[0]}")
    print(f"Number of Columns  : {df.shape[1]}")
    print("\nAvailable Raw Features:")
    for idx, col in enumerate(df.columns, 1):
        print(f"  {idx:2d}. {col}")
    print("="*60)
