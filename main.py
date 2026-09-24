import sys
import os

from src.data_loader import load_raw_dataset, display_load_info
from src.preprocessing import clean_and_preprocess_data
from src.eda import generate_eda_tables, generate_eda_charts
from src.random_forest_model import train_random_forest_model, display_model_summary
from src.prediction import predict_location_risk
from src.clustering import perform_spatial_clustering, view_accident_hotspots
from src.report import generate_complete_report

def display_menu():
    """
    Displays the main interactive terminal menu.
    """
    print("\n" + "="*50)
    print("      ACCIDENT HOTSPOT PREDICTION SYSTEM")
    print("="*50)
    print(" 1. Load Dataset")
    print(" 2. View Dataset Summary")
    print(" 3. Data Preprocessing & Feature Engineering")
    print(" 4. Exploratory Data Analysis (EDA)")
    print(" 5. Train Random Forest Model")
    print(" 6. Predict Accident Risk")
    print(" 7. Perform Spatial Clustering")
    print(" 8. View Accident Hotspots")
    print(" 9. Model Evaluation & Feature Importance")
    print("10. Generate Complete Project Report")
    print("11. Exit")
    print("="*50)

def main():
    raw_df = None
    cleaned_df = None
    model_results = None
    clustered_df = None
    loaded_path = ""

    while True:
        display_menu()
        choice = input("Enter your choice (1-11): ").strip()

        if choice == '1':
            try:
                raw_df, loaded_path = load_raw_dataset()
                display_load_info(raw_df, loaded_path)
            except Exception as e:
                print(f"\n[Error Loading Dataset] {str(e)}")

        elif choice == '2':
            if raw_df is None:
                print("\n[Notice] Dataset not loaded yet. Loading dataset automatically...")
                try:
                    raw_df, loaded_path = load_raw_dataset()
                except Exception as e:
                    print(f"[Error] {str(e)}")
                    continue
            
            if cleaned_df is None:
                cleaned_df = clean_and_preprocess_data(raw_df, verbose=False)

            print("\n" + "="*60)
            print("                 DATASET SUMMARY STATISTICS")
            print("="*60)
            print(f"Total Accident-Prone Locations : {len(cleaned_df)}")
            print(f"Total Accidents in 2023        : {int(cleaned_df['accidents_2023'].sum())}")
            print(f"Total Accidents in 2024        : {int(cleaned_df['accidents_2024'].sum())}")
            print(f"Total Accidents in 2025        : {int(cleaned_df['accidents_2025'].sum())}")
            print(f"Total Accidents Across All Yrs : {int(cleaned_df['total_accidents'].sum())}")
            print(f"Total Fatalities (2023 - 2025) : {int(cleaned_df['total_fatalities'].sum())}")
            
            top_loc_row = cleaned_df.sort_values(by='total_accidents', ascending=False).iloc[0]
            print(f"\nMaximum Accident Location    : {top_loc_row['location_name']}")
            print(f"                               ({int(top_loc_row['total_accidents'])} total accidents, {top_loc_row['zone']} Zone)")
            
            avg_acc = cleaned_df['total_accidents'].mean()
            print(f"Average Accidents per Location : {avg_acc:.2f}")
            print(f"Number of Zones                : {cleaned_df['zone'].nunique()} ({', '.join(cleaned_df['zone'].unique()[:4])}...)")
            print(f"Number of Police Stations      : {cleaned_df['police_station'].nunique()}")
            print("="*60)

        elif choice == '3':
            if raw_df is None:
                print("\n[Notice] Loading raw dataset...")
                raw_df, loaded_path = load_raw_dataset()
            
            cleaned_df = clean_and_preprocess_data(raw_df, verbose=True)

        elif choice == '4':
            if cleaned_df is None:
                if raw_df is None:
                    raw_df, loaded_path = load_raw_dataset()
                cleaned_df = clean_and_preprocess_data(raw_df, verbose=False)

            generate_eda_tables(cleaned_df)
            generate_eda_charts(cleaned_df)

        elif choice == '5':
            if cleaned_df is None:
                if raw_df is None:
                    raw_df, loaded_path = load_raw_dataset()
                cleaned_df = clean_and_preprocess_data(raw_df, verbose=False)

            model_results = train_random_forest_model(cleaned_df, verbose=True)

        elif choice == '6':
            if cleaned_df is None:
                if raw_df is None:
                    raw_df, loaded_path = load_raw_dataset()
                cleaned_df = clean_and_preprocess_data(raw_df, verbose=False)

            if model_results is None:
                print("\n[Notice] Training Random Forest model before prediction...")
                model_results = train_random_forest_model(cleaned_df, verbose=False)

            predict_location_risk(model_results, cleaned_df)

        elif choice == '7':
            if cleaned_df is None:
                if raw_df is None:
                    raw_df, loaded_path = load_raw_dataset()
                cleaned_df = clean_and_preprocess_data(raw_df, verbose=False)

            clustered_df = perform_spatial_clustering(cleaned_df)

        elif choice == '8':
            if cleaned_df is None:
                if raw_df is None:
                    raw_df, loaded_path = load_raw_dataset()
                cleaned_df = clean_and_preprocess_data(raw_df, verbose=False)

            view_accident_hotspots(cleaned_df)

        elif choice == '9':
            if cleaned_df is None:
                if raw_df is None:
                    raw_df, loaded_path = load_raw_dataset()
                cleaned_df = clean_and_preprocess_data(raw_df, verbose=False)

            if model_results is None:
                model_results = train_random_forest_model(cleaned_df, verbose=False)
            
            display_model_summary(model_results)

        elif choice == '10':
            if cleaned_df is None:
                if raw_df is None:
                    raw_df, loaded_path = load_raw_dataset()
                cleaned_df = clean_and_preprocess_data(raw_df, verbose=False)

            if model_results is None:
                model_results = train_random_forest_model(cleaned_df, verbose=False)

            generate_complete_report(cleaned_df, model_results, clustered_df)

        elif choice == '11':
            print("\nExiting Accident Hotspot Prediction System. Goodbye!\n")
            sys.exit(0)

        else:
            print("\n[Invalid Input] Please enter a choice between 1 and 11.")

if __name__ == "__main__":
    main()
