import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive background plot generation
import matplotlib.pyplot as plt
import seaborn as sns

CHARTS_DIR = os.path.join("outputs", "charts")

def ensure_charts_dir():
    if not os.path.exists(CHARTS_DIR):
        os.makedirs(CHARTS_DIR, exist_ok=True)

def generate_eda_tables(df):
    """
    Prints text-based ASCII tables for exploratory data analysis.
    """
    print("\n" + "="*75)
    print("                     EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*75)

    # Table 1: Yearly Accident Statistics
    print("\n[1] YEARLY ACCIDENT & FATALITY SUMMARY")
    print("-" * 65)
    print(f"{'Year':<10} | {'Total Accidents':<18} | {'Total Fatalities':<18} | {'Fatality Rate':<12}")
    print("-" * 65)
    acc_23, fat_23 = df['accidents_2023'].sum(), df['fatalities_2023'].sum()
    acc_24, fat_24 = df['accidents_2024'].sum(), df['fatalities_2024'].sum()
    acc_25, fat_25 = df['accidents_2025'].sum(), df['fatalities_2025'].sum()
    tot_acc, tot_fat = df['total_accidents'].sum(), df['total_fatalities'].sum()

    print(f"{'2023':<10} | {acc_23:<18.0f} | {fat_23:<18.0f} | {(fat_23/max(acc_23,1)):<12.2%}")
    print(f"{'2024':<10} | {acc_24:<18.0f} | {fat_24:<18.0f} | {(fat_24/max(acc_24,1)):<12.2%}")
    print(f"{'2025':<10} | {acc_25:<18.0f} | {fat_25:<18.0f} | {(fat_25/max(acc_25,1)):<12.2%}")
    print("-" * 65)
    print(f"{'TOTAL':<10} | {tot_acc:<18.0f} | {tot_fat:<18.0f} | {(tot_fat/max(tot_acc,1)):<12.2%}")
    print("-" * 65)

    # Table 2: Top 10 Accident-Prone Locations
    print("\n[2] TOP 10 ACCIDENT-PRONE LOCATIONS (BY TOTAL ACCIDENTS)")
    print("-" * 75)
    print(f"{'Rank':<4} | {'Location Name':<35} | {'Zone':<14} | {'Accidents':<9} | {'Fatalities':<10} | {'Risk':<6}")
    print("-" * 75)
    top_acc = df.sort_values(by=['total_accidents', 'total_fatalities'], ascending=False).head(10)
    for rank, (_, row) in enumerate(top_acc.iterrows(), 1):
        loc = (row['location_name'][:32] + '...') if len(str(row['location_name'])) > 35 else row['location_name']
        print(f"{rank:<4d} | {loc:<35} | {row['zone']:<14} | {int(row['total_accidents']):<9d} | {int(row['total_fatalities']):<10d} | {row['risk_level']:<6}")
    print("-" * 75)

    # Table 3: Top 10 Locations by Fatalities
    print("\n[3] TOP 10 LOCATIONS BY FATALITIES")
    print("-" * 75)
    print(f"{'Rank':<4} | {'Location Name':<35} | {'Zone':<14} | {'Fatalities':<10} | {'Accidents':<9} | {'Risk':<6}")
    print("-" * 75)
    top_fat = df.sort_values(by=['total_fatalities', 'total_accidents'], ascending=False).head(10)
    for rank, (_, row) in enumerate(top_fat.iterrows(), 1):
        loc = (row['location_name'][:32] + '...') if len(str(row['location_name'])) > 35 else row['location_name']
        print(f"{rank:<4d} | {loc:<35} | {row['zone']:<14} | {int(row['total_fatalities']):<10d} | {int(row['total_accidents']):<9d} | {row['risk_level']:<6}")
    print("-" * 75)

    # Table 4: Accidents & Fatalities by Zone
    print("\n[4] ACCIDENT DISTRIBUTION BY ZONE")
    print("-" * 75)
    print(f"{'Zone':<16} | {'Hotspots':<8} | {'Total Acc':<9} | {'Total Fat':<9} | {'Avg Acc/Loc':<12} | {'Avg Fat/Loc':<12}")
    print("-" * 75)
    zone_grp = df.groupby('zone').agg(
        hotspots=('sl_no', 'count'),
        total_acc=('total_accidents', 'sum'),
        total_fat=('total_fatalities', 'sum'),
        avg_acc=('total_accidents', 'mean'),
        avg_fat=('total_fatalities', 'mean')
    ).reset_index().sort_values(by='total_acc', ascending=False)
    
    for _, r in zone_grp.iterrows():
        print(f"{r['zone']:<16} | {int(r['hotspots']):<8d} | {int(r['total_acc']):<9d} | {int(r['total_fat']):<9d} | {r['avg_acc']:<12.2f} | {r['avg_fat']:<12.2f}")
    print("-" * 75)

    # Table 5: Risk Level Distribution
    print("\n[5] RISK CATEGORY DISTRIBUTION")
    print("-" * 45)
    print(f"{'Risk Category':<15} | {'Locations':<10} | {'Percentage':<10}")
    print("-" * 45)
    risk_counts = df['risk_level'].value_counts()
    for cat in ['LOW', 'MEDIUM', 'HIGH']:
        cnt = risk_counts.get(cat, 0)
        pct = (cnt / len(df)) * 100
        print(f"{cat:<15} | {cnt:<10d} | {pct:<9.1f}%")
    print("-" * 45)
    print("="*75)

def generate_eda_charts(df):
    """
    Generates and saves visual charts in outputs/charts/ directory.
    """
    ensure_charts_dir()
    sns.set_theme(style="whitegrid")
    chart_paths = []

    # Chart 1: Yearly Comparison
    plt.figure(figsize=(8, 5))
    years = ['2023', '2024', '2025']
    accidents = [df['accidents_2023'].sum(), df['accidents_2024'].sum(), df['accidents_2025'].sum()]
    fatalities = [df['fatalities_2023'].sum(), df['fatalities_2024'].sum(), df['fatalities_2025'].sum()]
    
    x = np.arange(len(years))
    width = 0.35
    
    plt.bar(x - width/2, accidents, width, label='Accidents', color='#1f77b4')
    plt.bar(x + width/2, fatalities, width, label='Fatalities', color='#d62728')
    
    plt.title('Hyderabad City: Accidents vs Fatalities (2023 - 2025)', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(x, years)
    plt.legend()
    plt.tight_layout()
    c1 = os.path.join(CHARTS_DIR, "accidents_by_year.png")
    plt.savefig(c1, dpi=300)
    plt.close()
    chart_paths.append(c1)

    # Chart 2: Top Hotspots
    plt.figure(figsize=(10, 6))
    top10 = df.sort_values(by='total_accidents', ascending=True).tail(10)
    short_names = [name[:30] + '...' if len(name) > 30 else name for name in top10['location_name']]
    
    plt.barh(short_names, top10['total_accidents'], color='#ff7f0e')
    plt.title('Top 10 High Accident Locations in Hyderabad', fontsize=14, fontweight='bold')
    plt.xlabel('Total Accidents (2023-2025)', fontsize=12)
    plt.tight_layout()
    c2 = os.path.join(CHARTS_DIR, "top_hotspots.png")
    plt.savefig(c2, dpi=300)
    plt.close()
    chart_paths.append(c2)

    # Chart 3: Accidents by Zone
    plt.figure(figsize=(9, 5))
    zone_totals = df.groupby('zone')['total_accidents'].sum().sort_values(ascending=False)
    sns.barplot(x=zone_totals.index, y=zone_totals.values, palette='viridis')
    plt.title('Total Accidents by Zone in Hyderabad City', fontsize=14, fontweight='bold')
    plt.xlabel('Zone', fontsize=12)
    plt.ylabel('Total Accidents', fontsize=12)
    plt.xticks(rotation=30)
    plt.tight_layout()
    c3 = os.path.join(CHARTS_DIR, "zone_accidents.png")
    plt.savefig(c3, dpi=300)
    plt.close()
    chart_paths.append(c3)

    # Chart 4: Risk Distribution
    plt.figure(figsize=(6, 6))
    risk_counts = df['risk_level'].value_counts()
    colors = {'LOW': '#2ca02c', 'MEDIUM': '#ff7f0e', 'HIGH': '#d62728'}
    ordered_colors = [colors[cat] for cat in risk_counts.index]
    
    plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=140, colors=ordered_colors)
    plt.title('Accident Risk Level Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    c4 = os.path.join(CHARTS_DIR, "risk_distribution.png")
    plt.savefig(c4, dpi=300)
    plt.close()
    chart_paths.append(c4)

    print("\nVisual EDA charts generated successfully:")
    for path in chart_paths:
        print(f"  [Saved] {path}")
