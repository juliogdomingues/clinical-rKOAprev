"""Data preparation: the single entry point that turns the raw ELSA CSV into
the analysis-ready knee-level dataset.

``load_and_prep_data`` reshapes wide participant rows into two knee-rows each,
derives the ``oa_knee`` outcome (KL>=2 or definite PF OA; arthroplasty dropped),
codes the binary history/symptom items (missing -> 0 via ``get_bin``), median-
targets continuous predictors for later in-fold imputation, expands categoricals
to dummies, and writes the data-prep audit CSVs. See docs/METHODOLOGY.md sec 2.
"""

import os
import numpy as np
import pandas as pd
from . import reporting as oarsi_reporting

# Global constant for exclusion (updated by user config potentially, but defaults here)
EXCLUDE_IDELSA: list[str] = []

def stress_test_all_vars(df, id_col='idelsa', outcome_cols=['oa_knee', 'kl', 'oapf']):
    """
    Tests what happens if we drop rows for ANY missing value in the entire dataset.
    """
    print("\n[STRESS TEST] Complete Case Analysis (All Variables)")

    feature_cols = [c for c in df.columns if c not in [id_col] + outcome_cols]
    rows_with_missing = df[feature_cols].isna().any(axis=1)

    n_total = len(df)
    n_dropped = rows_with_missing.sum()
    n_survivors = n_total - n_dropped
    pct_loss = (n_dropped / n_total) * 100

    print(f"   - Original Sample: {n_total} knees")
    print(f"   - Survivors (Complete Cases): {n_survivors}")
    print(f"   - LOST (Dropped): {n_dropped} ({pct_loss:.1f}%)")

    print(f"\n   [Top 10 Variables causing data loss]")
    missing_counts = df[feature_cols].isna().sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        print("      (None! Your dataset is perfectly complete.)")
    else:
        for var, count in missing_counts.head(10).items():
            pct = (count / n_total) * 100
            print(f"      - {var}: missing in {count} rows ({pct:.1f}%)")

    return missing_counts

def clean_kl(kl):
    return kl if kl in [0, 1, 2, 3, 4] else np.nan

def clean_oa(val):
    return val if val in [0, 1] else np.nan

def clean_sex(val):
    try:
        val = int(val)
        if val == 2: return 1 # Fem
        if val == 1: return 0 # Masc
        return np.nan  # numeric but out of domain (e.g. 3, 9) -> missing, not None
    except:
        return np.nan

def _audit_and_coerce(series: pd.Series, varname: str, *, valid_values: set[int] | None = None, audit_rows: list[dict] | None = None) -> pd.Series:
    """
    Coerce to numeric, turn '.D'/non-numeric/out-of-domain into NaN, and append audit info.
    """
    if audit_rows is None:
        audit_rows = []

    s_raw = series

    # Detect ".D" (string)
    s_str = s_raw.astype("string")
    dotd_mask = s_str.str.strip().str.upper().eq(".D")

    # Coerce
    s_num = pd.to_numeric(s_raw, errors="coerce")

    # Non-numeric (excluding true missing and ".D")
    nonnum_mask = s_raw.notna() & ~dotd_mask.fillna(False) & s_num.isna()

    # Out of domain
    if valid_values is not None:
        outdom_mask = s_num.notna() & ~s_num.isin(list(valid_values))
    else:
        outdom_mask = pd.Series(False, index=s_num.index)

    s_clean = s_num.where(~outdom_mask, np.nan)

    audit_rows.append(
        {
            "variable": varname,
            "n_total": int(len(s_raw)),
            "n_missing_original": int(pd.isna(s_raw).sum()),
            "n_dotD_string": int(dotd_mask.fillna(False).sum()),
            "n_non_numeric_to_nan": int(nonnum_mask.sum()),
            "n_out_of_domain_to_nan": int(outdom_mask.sum()),
            "n_missing_after_clean": int(pd.isna(s_clean).sum()),
            "n_new_nans_introduced": int(pd.isna(s_clean).sum() - pd.isna(s_raw).sum()),
            "valid_values": "" if valid_values is None else ",".join(map(str, sorted(valid_values))),
        }
    )
    return s_clean

def _coerce_and_mask_invalid(series: pd.Series, valid_values: set[int]) -> pd.Series:
    # kept for backward compatibility; prefer _audit_and_coerce where possible
    s = pd.to_numeric(series, errors="coerce")  # ".D" vira NaN
    s = s.where(s.isna() | s.isin(list(valid_values)), np.nan)
    return s

def load_and_prep_data(csv_path: str, outdir: str = './results_final_analysis', exclude_idelsa: list[str] = None):
    if exclude_idelsa is None:
        exclude_idelsa = EXCLUDE_IDELSA

    os.makedirs(outdir, exist_ok=True)
    print(f"[1/5] Carregando dados de: {csv_path}")
    df = pd.read_csv(csv_path)

    # Merge the REVISED radiographic readings (KL for both TF and PF compartments)
    # from the complementary Stata file, on idelsa. These supply the outcome:
    #   b_klpad/b_klpae = revised tibiofemoral KL (PA view)
    #   b_klpd /b_klpe  = patellofemoral KL (Perfil view)
    # If the file is absent we fall back to the legacy columns already in df.
    from .config import COMP_KL_DTA
    _rev_cols = ["b_klpad", "b_klpae", "b_klpd", "b_klpe"]
    # Socioeconomic predictors also live in the complementary file:
    #   a_escolar/b_escolar  = participant education (ordinal 1-4, waves 1/2)
    #   a_escolarmae         = mother's education (ordinal 1-4)
    #   b_vifb43_pmcat       = family monthly income, category midpoint (continuous)
    #   b_rendapercapita     = per-capita income (continuous)
    _ses_cols = ["a_escolar", "a_escolarmae", "b_escolar", "b_vifb43_pmcat", "b_rendapercapita"]
    if COMP_KL_DTA.exists():
        comp = pd.read_stata(str(COMP_KL_DTA))
        keep = ["idelsa"] + [c for c in _rev_cols + _ses_cols if c in comp.columns]
        df = df.merge(comp[keep].drop_duplicates(subset="idelsa"), on="idelsa", how="left")
        print(f"      merged from {COMP_KL_DTA.name}: {keep[1:]}")
    else:
        print(f"      WARNING: {COMP_KL_DTA} not found; using legacy TF-KL + binary-PF-OA outcome")

    audit_rows: list[dict] = []
    row_drop_rows: list[dict] = []
    audited_cols: set[str] = set()

    # ---- QA counters (raw) ----
    n_participants_raw = int(df['idelsa'].nunique()) if 'idelsa' in df.columns else int(len(df))
    n_rows_raw = int(len(df))

    # Participant exclusion
    if exclude_idelsa and "idelsa" in df.columns:
        mask_excl = df["idelsa"].isin(exclude_idelsa)
        n_excl_rows_raw = int(mask_excl.sum())
        n_before = int(len(df))
        df = df.loc[~mask_excl].copy()
        row_drop_rows.append(
            {
                "stage": "exclude_participants_config(raw_wide)",
                "n_before": n_before,
                "n_dropped": n_excl_rows_raw,
                "n_after": int(len(df)),
                "excluded_ids": ",".join(map(str, exclude_idelsa)),
            }
        )

    # --- Mapeamento Joelho (Completo) ---
    # 'kl' holds the tibiofemoral KL grade; 'oapf' holds the patellofemoral
    # signal. With the revised readings both are KL grades (0-4); PF_IS_KL=True.
    # Legacy fallback: 'kl' = original TF KL, 'oapf' = binary PF-OA (0/1).
    PF_IS_KL = ("b_klpd" in df.columns) and ("b_klpe" in df.columns)
    tf_right = "b_klpad" if "b_klpad" in df.columns else "b_kld"
    tf_left = "b_klpae" if "b_klpae" in df.columns else "b_kle"
    pf_right = "b_klpd" if PF_IS_KL else "b_oapfd"
    pf_left = "b_klpe" if PF_IS_KL else "b_oapfe"
    print(f"      outcome sources -> TF:({tf_right},{tf_left}) PF:({pf_right},{pf_left}) PF_IS_KL={PF_IS_KL}")

    right_map = {
        'kl': tf_right, 'oapf': pf_right,
        'womac_total': 'WOMTOTD_LB', 'womac_pain': 'WOMDORD_LB',
        'womac_stiffness': 'WOMRIGD_LB', 'womac_function': 'WOMFUND_LB'
    }
    left_map = {
        'kl': tf_left, 'oapf': pf_left,
        'womac_total': 'WOMTOTE_LB', 'womac_pain': 'WOMDORE_LB',
        'womac_stiffness': 'WOMRIGE_LB', 'womac_function': 'WOMFUNE_LB'
    }

    rows = []
    for side, mp in [('D', right_map), ('E', left_map)]:
        tmp = pd.DataFrame({'idelsa': df['idelsa'], 'side': side})
        for target, source in mp.items():
            tmp[target] = df.get(source, np.nan)

        target_vals = [2, 3] if side == 'D' else [1, 3]

        def get_bin(col):
            if col in df.columns:
                if col not in audited_cols:
                    _ = _audit_and_coerce(df[col], col, valid_values={0, 1, 2, 3}, audit_rows=audit_rows)
                    audited_cols.add(col)

                s = pd.to_numeric(df[col], errors="coerce")      # ".D" -> NaN
                s = s.where(s.isin([0, 1, 2, 3]), np.nan)        # fora do domínio -> NaN
                return s.isin(target_vals).astype(int)           # NaN -> False -> 0
            return np.zeros(len(df), dtype=int)

        tmp['history_surgery'] = get_bin('FDR3a_LB')
        tmp['history_trauma'] = get_bin('FDR2a_LB')
        tmp['frequent_symptoms'] = get_bin('DME47a_LB')
        tmp['recent_pain_7d'] = get_bin('DME51_2_LB')

        rows.append(tmp)

    long_df = pd.concat(rows, ignore_index=True)

    # ---- QA counters (long) ----
    n_rows_long = int(len(long_df))
    n_knees_long = n_rows_long
    n_participants_long = int(long_df['idelsa'].nunique())

    # --- Variáveis do Participante (LISTA COMPLETA) ---
    cols_needed = {
        # Demográficos
        'idadeb': 'age', 'b_imc1': 'bmi', 'rcta8': 'sex_raw',
        'vifa29': 'race_raw', 'a_nat_todos': 'occupation',

        # Antropometria
        'b_obesidadeabdominal': 'abdominal_obesity', 'b_rcq': 'waist_hip_ratio',
        'biob11': 'bone_mineral_content_kg', 'biob130': 'mineral_mass_kg', 'biob27': 'skeletal_muscle_mass_kg',

        # Testes Físicos e Incapacidade
        'TAL_MEAN2TRIALSSG_LB': 'sit_stand_test',
        'DME49_LB': 'knee_disability',

        # Estilo de Vida
        'b_fumante': 'smoking_status',
        'b_ativfisica': 'physical_activity_ipaq',
        'b_binge': 'alcohol_binge',
        'b_bebexcessivo': 'alcohol_excessive',
        'b_usodealcool': 'alcohol_use',
        'FDR5_LB': 'occ_stairs',
        'FDR6_LB': 'occ_kneeling', 'FDR7_LB': 'occ_squatting',
        'FDR4_LB': 'family_history_knee_replacement', # [NEW] Added

        # Metabólicos e Risco CV
        'b_has2_2': 'hypertension', 'b_dm_3': 'diabetes',
        'b_smj_as': 'metabolic_syndrome_JIS',
        'b_smj_an': 'metabolic_syndrome_NCEP',
        'b_smj_eu': 'metabolic_syndrome_IDF',
        'b_hipertrig': 'hypertriglyceridemia',
        'b_hipertrigmed': 'hypertrig_meds',
        'b_baixohdl': 'low_hdl',
        'b_baixohdlmed': 'low_hdl_meds',

        # Framingham
        'b_framingham_chd_chol_2': 'framingham_chd_chol',
        'b_framingham_chd_ldl_2': 'framingham_chd_ldl',
        'b_framingham_cvd_model1_2': 'framingham_cvd_model1',
        'b_framingham_cvd_model2_2': 'framingham_cvd_model2',

        # Socioeconômicos (do arquivo complementar). Educação = ordinal 1-4
        # (tratada como categórica/dummy, como as demais categóricas); renda =
        # contínua (imputada pela mediana no fold).
        'a_escolar': 'education_w1',
        'b_escolar': 'education_w2',
        'a_escolarmae': 'education_mother',
        'b_vifb43_pmcat': 'income_family_midpoint',
        'b_rendapercapita': 'income_per_capita',
    }

    # Busca segura (case insensitive)
    lower_cols = {c.lower(): c for c in df.columns}
    subset_cols = []
    rename_map = {}

    for key, new_name in cols_needed.items():
        if key.lower() in lower_cols:
            real_col = lower_cols[key.lower()]
            subset_cols.append(real_col)
            rename_map[real_col] = new_name

    part_df = df[['idelsa'] + subset_cols].drop_duplicates(subset='idelsa')
    part_df = part_df.rename(columns=rename_map)

    # Audit + saneamento explícito conforme dicionário
    if "physical_activity_ipaq" in part_df.columns:
        part_df["physical_activity_ipaq"] = _audit_and_coerce(part_df["physical_activity_ipaq"], "b_ativfisica", valid_values={1, 2, 3}, audit_rows=audit_rows)
    if "smoking_status" in part_df.columns:
        part_df["smoking_status"] = _audit_and_coerce(part_df["smoking_status"], "b_fumante", valid_values={0, 1, 2}, audit_rows=audit_rows)
    if "alcohol_use" in part_df.columns:
        part_df["alcohol_use"] = _audit_and_coerce(part_df["alcohol_use"], "b_usodealcool", valid_values={0, 1, 2}, audit_rows=audit_rows)
    if "alcohol_binge" in part_df.columns:
        part_df["alcohol_binge"] = _audit_and_coerce(part_df["alcohol_binge"], "B_BINGE/b_binge", valid_values={0, 1}, audit_rows=audit_rows)
    if "alcohol_excessive" in part_df.columns:
        part_df["alcohol_excessive"] = _audit_and_coerce(part_df["alcohol_excessive"], "b_bebexcessivo", valid_values={0, 1}, audit_rows=audit_rows)
    if "knee_disability" in part_df.columns:
        part_df["knee_disability"] = _audit_and_coerce(part_df["knee_disability"], "DME49_LB", valid_values={0, 1}, audit_rows=audit_rows)
    if "occ_stairs" in part_df.columns:
        part_df["occ_stairs"] = _audit_and_coerce(part_df["occ_stairs"], "FDR5_LB", valid_values={0, 1}, audit_rows=audit_rows)
    if "occ_kneeling" in part_df.columns:
        part_df["occ_kneeling"] = _audit_and_coerce(part_df["occ_kneeling"], "FDR6_LB", valid_values={0, 1}, audit_rows=audit_rows)
    if "occ_squatting" in part_df.columns:
        part_df["occ_squatting"] = _audit_and_coerce(part_df["occ_squatting"], "FDR7_LB", valid_values={0, 1}, audit_rows=audit_rows)
    if "family_history_knee_replacement" in part_df.columns: # [NEW] Audit
        part_df["family_history_knee_replacement"] = _audit_and_coerce(part_df["family_history_knee_replacement"], "FDR4_LB", valid_values={0, 1}, audit_rows=audit_rows)

    # Sexo (audita coerção numérica)
    if 'sex_raw' in part_df.columns:
        _ = _audit_and_coerce(part_df['sex_raw'], "rcta8(sex_raw)", valid_values={1, 2}, audit_rows=audit_rows)
        part_df['sex_female'] = part_df['sex_raw'].apply(clean_sex)
        part_df = part_df.drop(columns=['sex_raw'])

    # Dummies para categóricas. Educação (ordinal 1-4) é tratada como
    # categórica, consistente com as demais (fumo, atividade, álcool).
    cols_to_dummy = ['race_raw', 'occupation', 'smoking_status',
                     'physical_activity_ipaq', 'alcohol_use',
                     'education_w1', 'education_w2', 'education_mother']

    for c in cols_to_dummy:
        if c in part_df.columns:
            part_df[c] = pd.to_numeric(part_df[c], errors="coerce").fillna(-1).astype(int).astype(str)
            dummies = pd.get_dummies(part_df[c], prefix=c)
            # Drop the missing-category dummy (e.g. race_raw_-1): the -1 encodes
            # "value was missing", so keeping it would let the model use
            # missingness itself as a predictor. Real categories are retained.
            dummies = dummies.drop(columns=[col for col in dummies.columns if col.endswith("_-1")], errors="ignore")
            part_df = pd.concat([part_df, dummies], axis=1)

    final_df = long_df.merge(part_df, on='idelsa', how='left')

    # Defaults
    oarsi_reporting._save_missing_outcome_ids(final_df, outdir, label="before_any_outcome_rules")

    df_before_rules = final_df.copy()

    # Coerção numérica
    final_df["kl_raw_num"] = pd.to_numeric(final_df["kl"], errors="coerce")
    final_df["oapf_raw_num"] = pd.to_numeric(final_df["oapf"], errors="coerce")

    # 1) Drop por artroplastia (code 6)
    dropped_code6_mask = (final_df["kl_raw_num"] == 6) | (final_df["oapf_raw_num"] == 6)
    n_before_6 = int(len(final_df))
    n_drop_6 = int(dropped_code6_mask.sum())

    if n_drop_6 > 0:
        row_drop_rows.append(
            {
                "stage": "drop_arthroplasty_code6(KL_or_OAPF_eq_6)",
                "n_before": n_before_6,
                "n_dropped": n_drop_6,
                "n_after": n_before_6 - n_drop_6,
            }
        )
        final_df = final_df.loc[~dropped_code6_mask].copy()

    # 2) Recode para domínios válidos. TF ('kl') is always a KL grade (0-4).
    # PF ('oapf') is a KL grade (0-4) with the revised readings, or a binary
    # PF-OA flag (0/1) in the legacy fallback.
    kl_num = final_df["kl_raw_num"]
    oapf_num = final_df["oapf_raw_num"]
    pf_domain = [0, 1, 2, 3, 4] if PF_IS_KL else [0, 1]
    final_df["kl"] = kl_num.where(kl_num.isna() | kl_num.isin([0, 1, 2, 3, 4]), np.nan)
    final_df["oapf"] = oapf_num.where(oapf_num.isna() | oapf_num.isin(pf_domain), np.nan)

    # (QA)
    oarsi_reporting._save_missing_outcome_ids(final_df, outdir, label="after_recode_before_missing_filter")

    # 3) Drop final: somente se ambos missing
    n_before_outcome_filter = int(len(final_df))
    n_participants_before_outcome_filter = int(final_df["idelsa"].nunique())

    outcome_missing_mask = final_df[["kl", "oapf"]].isna().all(axis=1)
    n_missing_outcome = int(outcome_missing_mask.sum())

    row_drop_rows.append(
        {
            "stage": "drop_missing_outcome(kl&oapf)_after_recode",
            "n_before": n_before_outcome_filter,
            "n_dropped": n_missing_outcome,
            "n_after": n_before_outcome_filter - n_missing_outcome,
        }
    )

    # NEW: contagens mutuamente exclusivas
    oarsi_reporting._save_outcome_exclusion_counts(
        df_before_rules=df_before_rules,
        df_after_drop6_before_recode=final_df,
        outcome_missing_mask_after_recode=outcome_missing_mask,
        outdir=outdir,
        label="outcome_exclusion_counts_mutually_exclusive",
    )

    final_df = final_df.dropna(subset=["kl", "oapf"], how="all")

    # NEW: counters AFTER outcome filter
    n_after_outcome_filter = int(len(final_df))
    n_participants_after_outcome_filter = int(final_df["idelsa"].nunique())

    # 4) Derivar OA. Revised readings: OA if either compartment is KL>=2
    # (tibiofemoral OR patellofemoral). Legacy: TF KL>=2 OR binary PF-OA==1.
    if PF_IS_KL:
        final_df["oa_knee"] = ((final_df["kl"] >= 2) | (final_df["oapf"] >= 2)).astype(int)
    else:
        final_df["oa_knee"] = ((final_df["kl"] >= 2) | (final_df["oapf"] == 1)).astype(int)

    # NEW: drop raw outcome helper columns
    final_df = final_df.drop(columns=[c for c in ["kl_raw_num", "oapf_raw_num"] if c in final_df.columns])

    # ---- Save audit outputs ----
    pd.DataFrame(audit_rows).to_csv(os.path.join(outdir, "data_value_exclusions_audit.csv"), index=False)
    pd.DataFrame(row_drop_rows).to_csv(os.path.join(outdir, "row_drop_reasons.csv"), index=False)

    # ---- QA summary ----
    qa_summary = pd.DataFrame(
        [
            {"stage": "raw_participants_file(before_exclusions)", "n_rows": n_rows_raw, "n_participants": n_participants_raw, "n_knees": np.nan, "n_dropped": np.nan},
            {"stage": "long_knees_before_merge(after_exclusions)", "n_rows": n_rows_long, "n_participants": n_participants_long, "n_knees": n_knees_long, "n_dropped": np.nan},
            {"stage": "after_merge_before_outcome_filter", "n_rows": n_before_outcome_filter, "n_participants": n_participants_before_outcome_filter, "n_knees": n_before_outcome_filter, "n_dropped": np.nan},
            {"stage": "dropped_missing_outcome(kl&oapf)", "n_rows": np.nan, "n_participants": np.nan, "n_knees": np.nan, "n_dropped": n_missing_outcome},
            {"stage": "final_used_dataset", "n_rows": n_after_outcome_filter, "n_participants": n_participants_after_outcome_filter, "n_knees": n_after_outcome_filter, "n_dropped": np.nan},
        ]
    )
    qa_summary.to_csv(os.path.join(outdir, "data_prep_summary.csv"), index=False)

    miss = final_df.isna().sum().sort_values(ascending=False)
    miss_df = (
        miss.rename("n_missing")
        .to_frame()
        .assign(pct_missing=lambda d: (d["n_missing"] / len(final_df)) * 100.0)
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    miss_df.to_csv(os.path.join(outdir, "data_missingness_after_prep.csv"), index=False)

    # NEW: explicit "drops by feature/outcome" style reports
    oarsi_reporting._save_drop_reports(final_df, id_col="idelsa", outcome_cols=["kl", "oapf"], outdir=outdir)

    # Sort by ID to ensure deterministic order for GroupKFold
    if 'idelsa' in final_df.columns:
        final_df = final_df.sort_values('idelsa').reset_index(drop=True)

    return final_df
