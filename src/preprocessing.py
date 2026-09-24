import pandas as pd
import numpy as np

def clean_and_preprocess_data(df_raw, verbose=True):
    """
    Cleans raw dataset, handles missing values, converts types, performs feature engineering,
    and assigns target risk labels based on a documented severity score rule.
    """
    df = df_raw.copy()

    # 1. String Cleaning
    string_cols = ['zone', 'police_station', 'road_info', 'location_name']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Standardize specific known police station variants
    if 'police_station' in df.columns:
        df['police_station'] = df['police_station'].replace({'OU Sity': 'O U Sity', 'O U Sity': 'O U Sity'})

    # 2. Numerical Conversion & Missing Values Handling
    num_cols = [
        'accidents_2023', 'accidents_2024', 'accidents_2025', 'total_accidents',
        'fatalities_2023', 'fatalities_2024', 'fatalities_2025', 'total_fatalities',
        'start_km', 'end_km'
    ]
    
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        else:
            df[c] = 0.0

    # 3. Deduplication
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_count = initial_len - len(df)

    # 4. Data Consistency Verification
    # Re-calculate total_accidents and total_fatalities from individual year columns to ensure accuracy
    df['total_accidents'] = df['accidents_2023'] + df['accidents_2024'] + df['accidents_2025']
    df['total_fatalities'] = df['fatalities_2023'] + df['fatalities_2024'] + df['fatalities_2025']

    # 5. Feature Engineering
    df['average_yearly_accidents'] = df['total_accidents'] / 3.0
    df['accident_growth'] = df['accidents_2025'] - df['accidents_2023']
    df['fatality_rate'] = df['total_fatalities'] / np.maximum(df['total_accidents'], 1.0)
    
    # Severity Score = Total Accidents + (1.5 * Total Fatalities)
    # Weighs loss of life higher than non-fatal accidents
    df['severity_score'] = df['total_accidents'] + (1.5 * df['total_fatalities'])
    
    # Stretch length in KM (where start and end KM are specified)
    df['stretch_length_km'] = (df['end_km'] - df['start_km']).abs()

    # 6. Target Variable Creation (Risk Level)
    # Clear quantitative rule based on Severity Score percentiles
    def assign_risk_level(sev):
        if sev <= 13.0:
            return 'LOW'
        elif sev <= 18.0:
            return 'MEDIUM'
        else:
            return 'HIGH'

    df['risk_level'] = df['severity_score'].apply(assign_risk_level)

    if verbose:
        print("\n" + "="*60)
        print("          DATA PREPROCESSING & FEATURE ENGINEERING")
        print("="*60)
        print("1. Text Standardization:")
        print("   - Trimmed whitespace and standardized jurisdiction names (e.g., 'OU Sity' -> 'O U Sity').")
        print(f"2. Duplicate Records Removed: {dedup_count}")
        print("3. Missing Values & Types:")
        print("   - Filled missing accident/fatality counts with 0.")
        print("   - Converted numerical columns to numeric types.")
        print("4. Feature Engineering Created:")
        print("   - total_accidents (Sum of 2023, 2024, 2025)")
        print("   - total_fatalities (Sum of 2023, 2024, 2025)")
        print("   - average_yearly_accidents (total_accidents / 3.0)")
        print("   - accident_growth (accidents_2025 - accidents_2023)")
        print("   - fatality_rate (total_fatalities / total_accidents)")
        print("   - severity_score (total_accidents + 1.5 * total_fatalities)")
        print("   - stretch_length_km (|end_km - start_km|)")
        print("\n5. Documented Target Risk Label Rule:")
        print("   --------------------------------------------------------")
        print("   Formula: Severity Score = Total Accidents + (1.5 * Fatalities)")
        print("   - LOW Risk    : Severity Score <= 13.0")
        print("   - MEDIUM Risk : 13.0 < Severity Score <= 18.0")
        print("   - HIGH Risk   : Severity Score > 18.0")
        print("   --------------------------------------------------------")
        print("\nRisk Level Category Distribution:")
        counts = df['risk_level'].value_counts()
        for cat in ['LOW', 'MEDIUM', 'HIGH']:
            cnt = counts.get(cat, 0)
            pct = (cnt / len(df)) * 100
            print(f"   - {cat:<7}: {cnt:2d} locations ({pct:5.1f}%)")
        print("="*60)

    return df

def prepare_ml_features(df):
    """
    Prepares feature matrix X and target vector y for Random Forest Classifier.
    """
    feature_cols = [
        'accidents_2023', 'accidents_2024', 'accidents_2025',
        'fatalities_2023', 'fatalities_2024', 'fatalities_2025',
        'average_yearly_accidents', 'accident_growth', 'fatality_rate', 'stretch_length_km'
    ]
    
    X_num = df[feature_cols].copy()
    
    # One-hot encode Zone and Police Station to capture spatial categorical factors
    zone_dummies = pd.get_dummies(df['zone'], prefix='zone', drop_first=False)
    
    X = pd.concat([X_num, zone_dummies], axis=1)
    y = df['risk_level']
    
    return X, y
