import os
import pandas as pd

REPORTS_DIR = os.path.join("outputs", "reports")

def ensure_reports_dir():
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_complete_report(df, model_results, clustered_df=None):
    """
    Generates complete project summary report in terminal and saves to text file.
    """
    ensure_reports_dir()
    
    report_lines = []
    
    def log(text=""):
        print(text)
        report_lines.append(text)

    log("\n" + "="*80)
    log("                       COMPLETE PROJECT ACADEMIC REPORT")
    log("="*80)
    log("PROJECT TITLE:")
    log("  Accident Hotspot Prediction Using Random Forest and Spatial Clustering")
    log("\nDATASET SUPPLIED:")
    log("  Actual Hyderabad City Accident Black-Spots Dataset (2023 - 2025)")
    log(f"  Total Blackspot Locations Analyzed : {len(df)}")
    log(f"  Total Accidents Recorded (3 Years)  : {int(df['total_accidents'].sum())}")
    log(f"  Total Fatalities Recorded (3 Years) : {int(df['total_fatalities'].sum())}")

    log("\n1. DATA PREPROCESSING & CLEANING:")
    log("  - Stripped multi-level headers and text banners from raw CSV.")
    log("  - Cleaned whitespace, standardized jurisdiction spellings ('OU Sity' -> 'O U Sity').")
    log("  - Handled missing numeric values with 0 and recast to integer/float data types.")
    log("  - Removed duplicate records.")

    log("\n2. FEATURE ENGINEERING & TARGET LABEL RULE:")
    log("  - Engineered Features: average_yearly_accidents, accident_growth, fatality_rate, severity_score, stretch_length_km.")
    log("  - Target Risk Label Rule (Documented Formula):")
    log("      Severity Score = Total Accidents + (1.5 * Total Fatalities)")
    log("      * LOW Risk    : Severity Score <= 13.0")
    log("      * MEDIUM Risk : 13.0 < Severity Score <= 18.0")
    log("      * HIGH Risk   : Severity Score > 18.0")

    log("\n3. RANDOM FOREST MODEL CONFIGURATION & METRICS:")
    if model_results is not None:
        log(f"  - Algorithm           : RandomForestClassifier (n_estimators=100)")
        log(f"  - Accuracy            : {model_results['accuracy']:.4f} ({model_results['accuracy']*100:.2f}%)")
        log(f"  - Precision           : {model_results['precision']:.4f} ({model_results['precision']*100:.2f}%)")
        log(f"  - Recall              : {model_results['recall']:.4f} ({model_results['recall']*100:.2f}%)")
        log(f"  - F1 Score            : {model_results['f1_score']:.4f} ({model_results['f1_score']*100:.2f}%)")
        log("\n  - Top 5 Feature Importances:")
        top5 = model_results['feature_importances'].head(5)
        for feat, val in top5.items():
            clean_feat = feat.replace('zone_', 'Zone: ')
            log(f"      * {clean_feat:<30} : {val:.4f} ({val*100:5.2f}%)")
    else:
        log("  - Model training pending. Run Option 5 to include metrics.")

    log("\n4. SPATIAL CLUSTERING & GEOGRAPHIC LIMITATIONS:")
    log("  - Latitude & Longitude columns are NOT present in the dataset.")
    log("  - Standard Compliance: No fake coordinates invented.")
    log("  - Implemented KMeans analytical feature-space clustering proxy to group hotspots by risk profile.")

    log("\n5. TOP HIGH-RISK ACCIDENT HOTSPOTS:")
    top_spots = df.sort_values(by='severity_score', ascending=False).head(5)
    for rank, (_, row) in enumerate(top_spots.iterrows(), 1):
        log(f"  {rank}. {row['location_name']} [{row['zone']} Zone] - Acc: {int(row['total_accidents'])}, Fat: {int(row['total_fatalities'])}, Risk: {row['risk_level']}")

    log("\n6. ACADEMIC CONCLUSION:")
    log("  The Random Forest classification model effectively categorizes accident risk based on historical")
    log("  frequency, fatality rate, and spatial zone indicators. Highways and key arterial junctions (e.g. NH-44,")
    log("  Gaganpahad, Shamshabad) exhibit the highest severity index, demanding prioritized traffic safety")
    log("  interventions such as speed monitoring, pedestrian overpasses, and improved junction layout.")
    log("="*80)

    # Save report to text file
    report_file = os.path.join(REPORTS_DIR, "accident_prediction_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[Saved] Complete report saved to: {report_file}")
