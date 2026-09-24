import pandas as pd
import numpy as np

def predict_location_risk(model_results, df):
    """
    Interactive terminal interface for predicting accident risk for a selected location or user inputs.
    """
    if model_results is None or 'model' not in model_results:
        print("\n[Error] Random Forest Model is not trained yet. Please select Option 5 first.")
        return

    rf_model = model_results['model']
    feature_names = model_results['feature_names']

    print("\n" + "="*60)
    print("             ACCIDENT RISK PREDICTION MODULE")
    print("="*60)
    print("1. Select an existing location from dataset")
    print("2. Enter custom location feature values")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == '1':
        predict_existing_location(df, rf_model, feature_names)
    elif choice == '2':
        predict_custom_inputs(rf_model, feature_names)
    else:
        print("[Invalid Choice] Returning to main menu.")

def predict_existing_location(df, rf_model, feature_names):
    """
    Predict risk for an existing location in dataset.
    """
    print("\nSelect a zone to browse locations:")
    zones = sorted(df['zone'].unique())
    for idx, z in enumerate(zones, 1):
        print(f"  {idx}. {z}")
    
    z_choice = input(f"Enter Zone number (1-{len(zones)}) or press Enter for all: ").strip()
    
    if z_choice.isdigit() and 1 <= int(z_choice) <= len(zones):
        sel_zone = zones[int(z_choice) - 1]
        filtered_df = df[df['zone'] == sel_zone].reset_index(drop=True)
    else:
        filtered_df = df.copy()

    print(f"\nAvailable Locations ({len(filtered_df)} total):")
    print("-" * 65)
    print(f"{'No.':<4} | {'Location Name':<42} | {'Zone':<12}")
    print("-" * 65)
    for idx, row in filtered_df.head(20).iterrows():
        loc_short = (row['location_name'][:40] + '..') if len(str(row['location_name'])) > 42 else row['location_name']
        print(f"{idx+1:<4d} | {loc_short:<42} | {row['zone']:<12}")
    print("-" * 65)
    
    loc_idx_str = input(f"\nEnter Location No. (1-{min(20, len(filtered_df))}): ").strip()
    if not loc_idx_str.isdigit() or not (1 <= int(loc_idx_str) <= len(filtered_df)):
        print("[Invalid Selection] Returning to main menu.")
        return

    row = filtered_df.iloc[int(loc_idx_str) - 1]
    
    # Construct feature vector matching feature_names
    feat_dict = {}
    feat_dict['accidents_2023'] = row['accidents_2023']
    feat_dict['accidents_2024'] = row['accidents_2024']
    feat_dict['accidents_2025'] = row['accidents_2025']
    feat_dict['fatalities_2023'] = row['fatalities_2023']
    feat_dict['fatalities_2024'] = row['fatalities_2024']
    feat_dict['fatalities_2025'] = row['fatalities_2025']
    feat_dict['average_yearly_accidents'] = row['average_yearly_accidents']
    feat_dict['accident_growth'] = row['accident_growth']
    feat_dict['fatality_rate'] = row['fatality_rate']
    feat_dict['stretch_length_km'] = row['stretch_length_km']

    # Zone one-hot encoding
    for fname in feature_names:
        if fname.startswith('zone_'):
            z_name = fname.replace('zone_', '')
            feat_dict[fname] = 1 if row['zone'] == z_name else 0

    X_single = pd.DataFrame([feat_dict])[feature_names]
    
    pred_class = rf_model.predict(X_single)[0]
    probabilities = rf_model.predict_proba(X_single)[0]
    class_idx = list(rf_model.classes_).index(pred_class)
    confidence = probabilities[class_idx] * 100

    print("\n" + "="*60)
    print("                PREDICTION RESULT")
    print("="*60)
    print(f"Location           : {row['location_name']}")
    print(f"Zone               : {row['zone']}")
    print(f"Police Station     : {row['police_station']}")
    print(f"Road Information   : {row['road_info']}")
    print(f"\nPREDICTED RISK     : {pred_class} RISK")
    print(f"Confidence Score   : {confidence:.2f}%")
    print("\nClass Probabilities:")
    for cls, prob in zip(rf_model.classes_, probabilities):
        print(f"  - {cls:<7} Risk : {prob*100:6.2f}%")
    
    print("\nInput Factors Summary:")
    print(f"  - Total Accidents (2023-2025) : {int(row['total_accidents'])}")
    print(f"  - Total Fatalities (2023-2025): {int(row['total_fatalities'])}")
    print(f"  - Calculated Severity Score  : {row['severity_score']:.2f}")
    print(f"  - Fatality Rate              : {row['fatality_rate']:.2%}")
    print("\n[Disclaimer]")
    print("  This prediction is a Random Forest model-based statistical risk classification.")
    print("  It assesses historical severity patterns and does not guarantee future events.")
    print("="*60)

def predict_custom_inputs(rf_model, feature_names):
    """
    Predict risk based on manually entered custom location features.
    """
    print("\nEnter Custom Location Features:")
    try:
        acc_23 = float(input("  - Number of accidents in 2023: ").strip() or 0)
        acc_24 = float(input("  - Number of accidents in 2024: ").strip() or 0)
        acc_25 = float(input("  - Number of accidents in 2025: ").strip() or 0)
        fat_23 = float(input("  - Number of fatalities in 2023: ").strip() or 0)
        fat_24 = float(input("  - Number of fatalities in 2024: ").strip() or 0)
        fat_25 = float(input("  - Number of fatalities in 2025: ").strip() or 0)
        stretch = float(input("  - Stretch length in KM (e.g. 0.5): ").strip() or 0.5)
    except ValueError:
        print("[Error] Invalid numerical input. Returning to main menu.")
        return

    tot_acc = acc_23 + acc_24 + acc_25
    tot_fat = fat_23 + fat_24 + fat_25
    avg_acc = tot_acc / 3.0
    growth = acc_25 - acc_23
    fat_rate = tot_fat / max(tot_acc, 1.0)
    sev_score = tot_acc + (1.5 * tot_fat)

    feat_dict = {
        'accidents_2023': acc_23,
        'accidents_2024': acc_24,
        'accidents_2025': acc_25,
        'fatalities_2023': fat_23,
        'fatalities_2024': fat_24,
        'fatalities_2025': fat_25,
        'average_yearly_accidents': avg_acc,
        'accident_growth': growth,
        'fatality_rate': fat_rate,
        'stretch_length_km': stretch
    }

    # Set all zone dummies to 0 by default
    for fname in feature_names:
        if fname.startswith('zone_'):
            feat_dict[fname] = 0

    X_single = pd.DataFrame([feat_dict])[feature_names]
    
    pred_class = rf_model.predict(X_single)[0]
    probabilities = rf_model.predict_proba(X_single)[0]
    class_idx = list(rf_model.classes_).index(pred_class)
    confidence = probabilities[class_idx] * 100

    print("\n" + "="*60)
    print("            CUSTOM LOCATION PREDICTION RESULT")
    print("="*60)
    print(f"PREDICTED ACCIDENT RISK : {pred_class} RISK")
    print(f"Confidence Score        : {confidence:.2f}%")
    print("\nCalculated Feature Metrics:")
    print(f"  - Total Accidents     : {int(tot_acc)}")
    print(f"  - Total Fatalities    : {int(tot_fat)}")
    print(f"  - Severity Score      : {sev_score:.2f}")
    print(f"  - Fatality Rate       : {fat_rate:.2%}")
    print("\nClass Probabilities:")
    for cls, prob in zip(rf_model.classes_, probabilities):
        print(f"  - {cls:<7} Risk : {prob*100:6.2f}%")
    print("="*60)
