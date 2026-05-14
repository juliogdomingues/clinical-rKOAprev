
import pandas as pd
import oarsi_data as p

# Definitions from run_comprehensive_comparison.py
BASE_EXCLUDE = [
    'idelsa', 'side', 'kl', 'oapf', 'oa_knee',
    'kl_raw_num', 'oapf_raw_num',
    'race_raw', 'occupation', 'smoking_status',
    'physical_activity_ipaq', 'alcohol_use'
]

BIO_VARS = [
    'bone_mineral_content_kg', 'mineral_mass_kg', 'skeletal_muscle_mass_kg', 
    'abdominal_obesity', 'waist_hip_ratio', 'bmi'
]

SYMPTOM_VARS = [
    'frequent_symptoms', 'recent_pain_7d', 'knee_disability', 
    'occ_stairs', 'occ_kneeling', 'occ_squatting' 
]

def main():
    print("Loading data...")
    df = p.load_and_prep_data("./base_stata/stataToCsvMG.csv")
    
    all_cols = [c for c in df.columns if c not in BASE_EXCLUDE]
    
    print(f"\nTotal Variables Considered: {len(all_cols)}\n")
    
    print("--- SYMPTOM VARIABLES (Used in 'With Symptoms' & 'Virtual Max') ---")
    for v in SYMPTOM_VARS:
        if v in all_cols: print(f"  - {v}")
        
    print("\n--- BIO-IMPEDANCE/ANTHRO VARIABLES (Used in 'Virtual Max') ---")
    for v in BIO_VARS:
        if v in all_cols: print(f"  - {v}")

    print("\n--- CLINICAL/DEMOGRAPHIC VARIABLES (Used in ALL models) ---")
    # Clinical is everything else
    clinical = [c for c in all_cols if c not in SYMPTOM_VARS and c not in BIO_VARS]
    for v in clinical:
        print(f"  - {v}")

if __name__ == "__main__":
    main()
