#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUPER PIPELINE FINAL v2: Knee OA Prediction (ALL VARIABLES INCLUDED)
--------------------------------------------------------------------
Refactored into modules:
  - oarsi_data.py
  - oarsi_analysis.py
  - oarsi_reporting.py
  - oarsi_utils.py
"""

import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Custom modules
import oarsi_data
import oarsi_analysis

warnings.filterwarnings('ignore')

# ================= CONFIGURAÇÃO =================
CSV_PATH = "./base_stata/stataToCsvMG.csv"

OUTDIR = './results_final_analysis'
os.makedirs(OUTDIR, exist_ok=True)
RND = 42
plt.style.use('seaborn-v0_8-whitegrid')

USE_WOMAC = False

# NEW: participant exclusions (drops both knees for each excluded idelsa)
EXCLUDE_IDELSA: list[str] = []
# EXCLUDE_IDELSA: list[str] = ["3509MG"]

# Pass configuration to modules if needed (or pass as args)
oarsi_data.EXCLUDE_IDELSA = EXCLUDE_IDELSA


# ================= EXECUÇÃO =================
if __name__ == "__main__":
    if CSV_PATH and os.path.exists(CSV_PATH):
        # 1. Load and Clean
        df = oarsi_data.load_and_prep_data(CSV_PATH, outdir=OUTDIR, exclude_idelsa=EXCLUDE_IDELSA)
        
        # 2. Stress Test
        oarsi_data.stress_test_all_vars(df)
        
        # 3. Analyze
        res = oarsi_analysis.run_analysis(df, outdir=OUTDIR, use_womac=USE_WOMAC)
        
        res.to_csv(os.path.join(OUTDIR, 'final_comparison_table.csv'), index=False)
        print("\n=== RESULTADOS ===")
        print(res[['Model', 'AUC', 'Std']])
        print(f"\nArquivos salvos em: {OUTDIR}")
    else:
        print(f"CSV não encontrado: {CSV_PATH}")


# === HOW TO RUN IT ===
# Place this right after you load the data in the main block:
# df = load_and_prep_data(CSV_PATH)
# stress_test_all_vars(df)