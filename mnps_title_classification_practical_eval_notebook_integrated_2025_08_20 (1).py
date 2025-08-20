# MNPS Title Classification + Practical Evaluation (Integrated Notebook)
#
# Combines:
#  • Resource & prompt workflow from the Mini‑Hackathon starter
#  • Title‑mention hygiene + masking/leakage audit from Evaluation & Reliability
#  • Practical evaluation (confusion, severity, cost, SWCI) from Finalized Working Code
#
# **Evaluation stance:**
#  • PRIMARY method → Practical evaluation (confusion, severity & cost matrices, SWCI KPI) and review queues.
#  • SUPPORT method → Title accuracy with exact 95% CI (PASS if LB ≥ 0.90) for statistical credibility.
#
# Run order (first time):
#  1) Section 1 → 2 → 2.1  (setup & config)
#  2) Section 3  (load evaluation sample: New Sample_08.07.2025.csv)
#  3) Section 4  (resources; optional, helps the prompt)
#  4) Section 5  (prompt + schema; test on one row if you like)
#  5) Section 6  (batch run → model_outputs.csv)
#  6) Section 7  (adjudication sheet → give to reviewer)
#  7) [Human step] Reviewer completes MNPS_Adjudication_Filled.csv
#  8) Section 8  (**SUPPORT**: Title accuracy + exact 95% CI; writes support_metrics.json)
#  9) Section 9  (**PRIMARY**: Practical eval → confusion, severity, cost, SWCI; merges support metrics into summary)
# 10) Section 10 (exports & recap)
#
# On re‑runs after prompt tweaks: bump PROMPT_VERSION (Section 2), re‑run 6–10.

# =============================
# Section 1 — Environment Setup
# =============================
!pip -q install --upgrade openai pydantic

import os, re, json, math, glob, shutil, sys, subprocess, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

# Colab userdata API: if you set OPENAI_API_KEY there (🔑 icon), we use it.
try:
    from google.colab import userdata  # type: ignore
    os.environ.setdefault("OPENAI_API_KEY", userdata.get('OPENAI_API_KEY') or "")
except Exception:
    pass

assert os.environ.get("OPENAI_API_KEY"), (
    "Set OPENAI_API_KEY in Colab (left sidebar 🔑) or os.environ before running Section 5.")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ==========================
# Section 2 — Config & Paths
# ==========================
# Acceptance gates (used in practical eval); CI acceptance is set in Section 8.
ACCEPTABLE_MAJOR_ERROR_RATE = 0.02  # ≤ 2%
ACCEPTABLE_MINOR_ERROR_RATE = 0.05  # ≤ 5%

# Title acceptance rule (Section 8): PASS if exact 95% lower bound ≥ 0.90
ALPHA = 0.05
ACCEPT_LB = 0.90

# Labels
MAJOR_CLASSES = ["Technician", "Specialist", "Analyst", "Manager", "Coordinator", "Director", "Other"]
MINOR_CLASSES = ["I", "II", "III", "Lead"]
MINOR_IDX = {m:i for i,m in enumerate(MINOR_CLASSES)}

# Cost maps (annual $)
COST_MAP_MAJOR = {
    "Technician": 54225.00,
    "Specialist": 68765.00,
    "Analyst":    78208.00,
    "Manager":   103904.00,
    "Coordinator":123133.00,
    "Director":  146246.00,
    "Other":      73780.00,
}
COST_MAP_MINOR = {
    "I":  4405.00,
    "II": 3366.94,
    "III":6800.00,
    "Lead":6800.00,
}

# Severity grid (rows=true, cols=pred) for majors; minor‑only severity handled separately
SEVERITY_MAJOR_GRID = {
    ("Technician","Technician"):0, ("Technician","Specialist"):4, ("Technician","Analyst"):5, ("Technician","Manager"):6, ("Technician","Coordinator"):4, ("Technician","Director"):7, ("Technician","Other"):6,
    ("Specialist","Technician"):4, ("Specialist","Specialist"):0, ("Specialist","Analyst"):4, ("Specialist","Manager"):5, ("Specialist","Coordinator"):3, ("Specialist","Director"):6, ("Specialist","Other"):6,
    ("Analyst","Technician"):5, ("Analyst","Specialist"):4, ("Analyst","Analyst"):0, ("Analyst","Manager"):4, ("Analyst","Coordinator"):4, ("Analyst","Director"):5, ("Analyst","Other"):6,
    ("Manager","Technician"):6, ("Manager","Specialist"):5, ("Manager","Analyst"):4, ("Manager","Manager"):0, ("Manager","Coordinator"):5, ("Manager","Director"):4, ("Manager","Other"):6,
    ("Coordinator","Technician"):4, ("Coordinator","Specialist"):3, ("Coordinator","Analyst"):4, ("Coordinator","Manager"):5, ("Coordinator","Coordinator"):0, ("Coordinator","Director"):6, ("Coordinator","Other"):6,
    ("Director","Technician"):7, ("Director","Specialist"):6, ("Director","Analyst"):5, ("Director","Manager"):4, ("Director","Coordinator"):6, ("Director","Director"):0, ("Director","Other"):6,
    ("Other","Technician"):5, ("Other","Specialist"):5, ("Other","Analyst"):5, ("Other","Manager"):5, ("Other","Coordinator"):5, ("Other","Director"):5, ("Other","Other"):0,
}

# Optional: severity→cost amplification (toggle)
APPLY_SEVERITY_MULTIPLIER = True
SEVERITY_COST_MULTIPLIER = {0:0.00, 2:1.15, 3:1.35, 4:1.75, 5:2.50, 6:4.00, 7:6.00}

# Column names (practical eval expects these)
COL_RECORD_ID  = "record_id"
COL_TRUE_MAJOR = "true_major_role"
COL_PRED_MAJOR = "pred_major_role"
COL_TRUE_MINOR = "true_minor_role"
COL_PRED_MINOR = "pred_minor_role"

# Outputs & inputs
OUTPUT_DIR = "artifacts_integrated"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
MODEL_OUT_CSV = "model_outputs.csv"
ADJ_SHEET_CSV = "MNPS_Adjudication_Sheet.csv"
ADJ_FILLED_CSV = "MNPS_Adjudication_Filled.csv"  # reviewer returns this

# Prompt/model
PROMPT_VERSION = "v1"
MODEL_NAME = "gpt-4o"

# ================================
# Section 2.1 — Data Input Options
# ================================
# Option A: set a direct URL (kept from Evaluation & Reliability)
DATA_URL = None  # e.g., "https://raw.githubusercontent.com/.../New%20Sample_08.07.2025.csv"
# Option B: upload locally via Colab file picker (recommended for ad‑hoc)
LOCAL_INPUT_DIR = "local_inputs"; Path(LOCAL_INPUT_DIR).mkdir(exist_ok=True)

# Helper: robust CSV read (handles encodings)
def read_csv_robust(path: str) -> pd.DataFrame:
    attempts = [
        dict(encoding="utf-8"),
        dict(encoding="utf-8-sig"),
        dict(encoding="cp1252", engine="python"),
        dict(encoding="latin1", engine="python"),
    ]
    last = None
    for kw in attempts:
        try:
            return pd.read_csv(path, **kw)
        except Exception as e:
            last = e
    raise last or ValueError(f"Failed to read CSV: {path}")

# ======================================================
# Section 3 — Load Evaluation Sample (Only‑Exclusion Rule)
# ======================================================
from google.colab import files  # type: ignore

print("If you don't use DATA_URL, upload 'New Sample_08.07.2025.csv' now…")
_uploaded = files.upload() if not DATA_URL else {}
for name in _uploaded:
    shutil.move(name, str(Path(LOCAL_INPUT_DIR)/name))

if DATA_URL:
    df_eval = read_csv_robust(DATA_URL)
else:
    # pick first CSV in LOCAL_INPUT_DIR if multiple
    cands = sorted(Path(LOCAL_INPUT_DIR).glob("*.csv"))
    assert cands, f"No CSV uploaded into {LOCAL_INPUT_DIR}."
    df_eval = read_csv_robust(str(cands[0]))

# Only a priori exclusion: thin/empty Position Summary
MIN_SUMMARY_CHARS = 10
if "Position Summary" not in df_eval.columns:
    raise ValueError("Missing required 'Position Summary' column in evaluation CSV.")

df_eval["Position Summary"] = df_eval["Position Summary"].astype(str).str.strip()
df_eval = df_eval[df_eval["Position Summary"].str.len() > MIN_SUMMARY_CHARS].copy()

# Stable ID
if "Record_ID" not in df_eval.columns:
    df_eval.insert(0, "Record_ID", np.arange(1, len(df_eval)+1))

print(f"Evaluation rows loaded: n={len(df_eval)}")
df_eval.head(3)

# ======================================
# Section 4 — Resources (optional/assist)
# ======================================
# If you have the 2025u‑mnps‑minihackathon zip, unzip here to help the prompt.
RESOURCES_ZIP = None  # e.g., "/content/2025u-mnps-minihackathon.zip"
RESOURCES_DIR = None

if RESOURCES_ZIP and Path(RESOURCES_ZIP).is_file():
    !unzip -o "$RESOURCES_ZIP" -d "/content/"
    RESOURCES_DIR = "/content/2025u-mnps-minihackathon/prompt-resources/"

roles_lookup = None
ksac_table = None
try:
    if RESOURCES_DIR and Path(RESOURCES_DIR).exists():
        roles_lookup = pd.read_csv(Path(RESOURCES_DIR)/"MNPS Roles.csv")
        ksac_table   = pd.read_csv(Path(RESOURCES_DIR)/"MNPS KSACs.csv")
except Exception as e:
    print("Resource load warning:", e)

# ========================================
# Section 5 — Prompt, Hygiene & Schema (LLM)
# ========================================
zero_shot_prompt = (
    """
Objective: Evaluate and group jobs based on job functions (not titles). Classify each record into:
  • major_role_group ∈ {Specialist, Analyst, Director, Manager, Technician, Coordinator}
  • minor_sub_group ∈ {I, II, III, Lead}
Return a proposed MNPS‑style new job title and a concise justification anchored on duties/KSACs.

Process:
- Compare attributes: Position Summary, Essential Functions, Education, Work Experience, Licenses/Certifications, KSAs.
- Use MNPS Roles + KSACs (if provided) to guide grouping.
- Favor qualitative, holistic functional alignment with KSACs over keyword title matching.

Constraints:
1) Prioritize signals: Essential Functions > Education/Experience > Licensure/Certifications.
2) Restrict major_role_group to the allowed set.
3) Return confidence_0to1 as a conservative float in [0, 1].
4) Echo back the record_id provided.

Title‑Mention Hygiene (Leakage Control):
5) Ignore explicit job title strings in Position Summary / Essential Functions (e.g., “Director of …”). Treat as uninformative branding.
6) Internally rewrite such strings as “this role” and base decisions on duties/scope, Education/Experience, Licensure.
7) Do not quote those strings in justification; justify using duties, scope/leadership, credentials, KSAC alignment.
8) If duties/scope contradict an in‑text title, follow duties/scope/credentials.
"""
).strip()

class JobClassification(BaseModel):
    record_id: int = Field(..., description="Record_ID from the evaluation CSV.")
    job_title_original: str = Field(..., description="Original job title.")
    new_job_title: str = Field(..., description="Proposed MNPS‑style title.")
    major_role_group: str = Field(..., description="One of allowed majors.")
    minor_sub_group: str = Field(..., description="Level: I, II, III, Lead.")
    grouping_justification: str = Field(..., description="One‑sentence rationale using duties/credentials/KSACs.")
    confidence_0to1: float = Field(..., ge=0.0, le=1.0, description="Conservative confidence.")

class JobClassificationTable(BaseModel):
    job_classification_table: List[JobClassification]
    narrative_rationale: Optional[str] = None

# Title mention masking (detect + mask → leakage audit)
TITLE_TERMS = r"(senior|sr\.?|junior|jr\.?|lead|chief|principal|head|assistant|associate|executive\s+director|vice\s+president|vp|director|manager|specialist|analyst|technician|coordinator|officer|administrator|architect|engineer|supervisor)"

def mask_title_mentions(text: str):
    """Return masked text and a boolean leakage flag if a probable title mention was found."""
    if not isinstance(text, str) or not text:
        return text or "", False
    pattern = rf"(?i)\b(?:the\s+)?(?:{TITLE_TERMS})\b(?:\s*(?:of|for|,)?\s*[A-Z][\w/&\-, ]+)?"
    flagged = re.search(pattern, text) is not None
    masked = re.sub(pattern, "this role", text)
    return masked, flagged

# Quick single‑row test (optional)
if len(df_eval) > 0:
    r0 = df_eval.iloc[0]
    ps0, _ = mask_title_mentions(r0.get('Position Summary',''))
    ef0, _ = mask_title_mentions(r0.get('Essential Functions',''))
    user_payload_0 = f"""
Record_ID: {int(r0['Record_ID'])}
Original Job Title: {r0.get('Job Description Name','')}
Position Summary: {ps0}
Education: {r0.get('Education','')}
Work Experience: {r0.get('Work Experience','')}
Licenses and Certifications: {r0.get('Licenses and Certifications','')}
Knowledge, Skills and Abilities: {r0.get('Knowledge, Skills and Abilities','')}
Essential Functions: {ef0}
"""
    messages_0 = [
        {"role": "developer", "content": zero_shot_prompt},
        {"role": "user", "content": "Classify the following job description. Output 1 item in job_classification_table.\n" + user_payload_0}
    ]
    # (You can comment out this test call if you prefer.)
    # _resp0 = client.beta.chat.completions.parse(
    #     model=MODEL_NAME, messages=messages_0, temperature=0.2, max_tokens=1000,
    #     response_format=JobClassificationTable,
    # )
    # print("Schema OK; ready to batch.")

# ============================================
# Section 6 — Batch Model Run → model_outputs.csv
# ============================================

def classify_one(row) -> dict:
    ps, leak_ps = mask_title_mentions(row.get('Position Summary',''))
    ef, leak_ef = mask_title_mentions(row.get('Essential Functions',''))

    user_payload = f"""
Record_ID: {int(row['Record_ID'])}
Original Job Title: {row.get('Job Description Name','')}
Position Summary: {ps}
Education: {row.get('Education','')}
Work Experience: {row.get('Work Experience','')}
Licenses and Certifications: {row.get('Licenses and Certifications','')}
Knowledge, Skills and Abilities: {row.get('Knowledge, Skills and Abilities','')}
Essential Functions: {ef}
"""
    messages = [
        {"role": "developer", "content": zero_shot_prompt},
        {"role": "user", "content": "Classify the following job description. Output 1 item in job_classification_table.\n" + user_payload}
    ]

    resp = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=1000,
        response_format=JobClassificationTable,
    )
    parsed = resp.choices[0].message.parsed.job_classification_table[0].model_dump()

    return {
        "Record_ID": int(parsed["record_id"]),
        "Model_Prompt_Version": PROMPT_VERSION,
        "Model_Run_Timestamp": dt.datetime.utcnow().isoformat(),
        "Model_Pred_Title": parsed["new_job_title"],
        "Model_Pred_Major": parsed["major_role_group"],
        "Model_Pred_Minor": parsed["minor_sub_group"],
        "Model_Confidence_0to1": float(parsed["confidence_0to1"]),
        "Model_Rationale_Short": parsed["grouping_justification"],
        "Leakage_Flag": bool(leak_ps or leak_ef),
    }

pred_rows = [classify_one(r) for _, r in df_eval.iterrows()]
results_df = pd.DataFrame(pred_rows)
results_df.to_csv(MODEL_OUT_CSV, index=False)
print(f"Saved {MODEL_OUT_CSV} with {len(results_df)} rows.")
results_df.head(3)

# ==============================================
# Section 7 — Build Adjudication Sheet (Human)
# ==============================================

def _trim(x, n=1200):
    s = str(x or "")
    return (s[:n] + " …") if len(s) > n else s

context_cols = [
    "Record_ID",
    "Job Description Name",
    "Position Summary",
    "Education",
    "Work Experience",
    "Licenses and Certifications",
    "Knowledge, Skills and Abilities",
    "Essential Functions",
]
for c in context_cols:
    if c not in df_eval.columns:
        df_eval[c] = ""

adj = df_eval[context_cols].copy()
adj["Position Summary"] = adj["Position Summary"].map(lambda s: _trim(s, 1200))
adj["Essential Functions"] = adj["Essential Functions"].map(lambda s: _trim(s, 1200))

keep_pred_cols = [
    "Record_ID",
    "Model_Prompt_Version",
    "Model_Run_Timestamp",
    "Model_Pred_Title",
    "Model_Pred_Major",
    "Model_Pred_Minor",
    "Model_Confidence_0to1",
    "Model_Rationale_Short",
    "Leakage_Flag",
]
for c in keep_pred_cols:
    if c not in results_df.columns:
        results_df[c] = ""

sheet = adj.merge(results_df[keep_pred_cols], on="Record_ID", how="left")

# Blank columns the reviewer will complete
review_cols = {
    "Human_Title_Correct_YN": "",   # Y / N
    "Human_Major_Correct_YN": "",   # Y / N
    "Human_Minor_Correct_YN": "",   # Y / N
    "Gold_Title": "",               # fill if model title is wrong
    "Gold_Major": "",               # canonical major
    "Gold_Minor": "",               # canonical minor
    "Error_Category": "",           # e.g., Wrong Family, Wrong Seniority, KSAC Misweight, Licensure Misread, Title Leakage, Ambiguous, Other
    "Adjudication_Notes": "",
}
for k,v in review_cols.items():
    sheet[k] = v

sheet.to_csv(ADJ_SHEET_CSV, index=False)
print(f"Wrote adjudication sheet → {ADJ_SHEET_CSV}  (rows={len(sheet)})")
sheet.head(3)

# ==================================================================
# Section 8 — SUPPORT METHOD: Title Accuracy (Exact 95% CI) & By‑Source
# ==================================================================
# After the reviewer returns MNPS_Adjudication_Filled.csv next to this notebook.

if not Path(ADJ_FILLED_CSV).is_file():
    print(f"Upload the filled adjudication file as '{ADJ_FILLED_CSV}' and re‑run this cell.")
else:
    adjf = read_csv_robust(ADJ_FILLED_CSV)
    # Normalize Y/N columns
    def yn_to_bool(s):
        s = str(s).strip().upper()
        return True if s in {"Y", "YES", "TRUE", "1"} else False if s in {"N", "NO", "FALSE", "0"} else np.nan

    for c in ["Human_Title_Correct_YN","Human_Major_Correct_YN","Human_Minor_Correct_YN"]:
        if c in adjf.columns:
            adjf[c] = adjf[c].map(yn_to_bool)
        else:
            adjf[c] = np.nan

    # Title correctness
    if "Human_Title_Correct_YN" in adjf.columns and adjf["Human_Title_Correct_YN"].notna().any():
        title_correct = adjf["Human_Title_Correct_YN"].fillna(False).astype(bool)
    else:
        # fallback: compare Model_Pred_Title vs Gold_Title if present
        title_correct = adjf.get("Model_Pred_Title", pd.Series(index=adjf.index)).astype(str).str.strip() \
            == adjf.get("Gold_Title", pd.Series(index=adjf.index)).astype(str).str.strip()

    k = int(title_correct.sum()); n = int(len(adjf))

    # Exact (Clopper–Pearson) CI
    try:
        from scipy.stats import beta
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy"])  # install once
        from scipy.stats import beta

    def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
        if n == 0: return (0.0, 1.0)
        lb = 0.0 if k == 0 else beta.ppf(alpha/2, k, n-k+1)
        ub = 1.0 if k == n else beta.ppf(1 - alpha/2, k+1, n-k)
        return float(lb), float(ub)

    acc = k/n if n else float('nan')
    lb, ub = clopper_pearson_ci(k, n, alpha=ALPHA)
    decision = "PASS" if lb >= ACCEPT_LB else "REJECT"

    print(f"Title accuracy: {acc:.3f}  (exact 95% CI: [{lb:.3f}, {ub:.3f}])  → {decision}")

    # Optional: by Source (Internal/External) with Wilson 95% intervals
    def wilson_ci(k: int, n: int, z=1.96):
        if n == 0: return (0.0, 1.0)
        p = k/n
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        half = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
        return max(0.0, center-half), min(1.0, center+half)

    if "Source" in adjf.columns:
        rows = []
        for src, g in adjf.groupby("Source"):
            k_s = int((g.index.isin(title_correct[title_correct].index)).sum()) if isinstance(title_correct, pd.Series) else int(g["Human_Title_Correct_YN"].sum())
            n_s = int(len(g))
            wlb, wub = wilson_ci(k_s, n_s)
            rows.append({"Source": src, "n": n_s, "k": k_s, "acc": k_s/n_s if n_s else np.nan, "wilson_lb": wlb, "wilson_ub": wub})
        by_source = pd.DataFrame(rows)
        by_source.to_csv(Path(OUTPUT_DIR)/"results_by_source.csv", index=False)
        print("By‑source results →", Path(OUTPUT_DIR)/"results_by_source.csv")

    adjf.to_csv(Path(OUTPUT_DIR)/"MNPS_Scored_Detail.csv", index=False)

    # Persist support metrics for Section 9 to merge into the main summary
    support_metrics = {
        "title_accuracy": float(acc),
        "ci_95_lb": float(lb),
        "ci_95_ub": float(ub),
        "alpha": float(ALPHA),
        "accept_lb": float(ACCEPT_LB),
        "decision": decision,
        "k_correct": int(k),
        "n_total": int(n),
    }
    with open(Path(OUTPUT_DIR)/"support_metrics.json", "w", encoding="utf-8") as f:
        json.dump(support_metrics, f, indent=2)
    print("Support metrics →", Path(OUTPUT_DIR)/"support_metrics.json")

# =====================================================================
# Section 9 — PRIMARY METHOD: Practical Evaluation (Confusion, Severity, Cost, SWCI)
# =====================================================================
# Build the standard predictions table by merging adjudication (Gold) with model outputs.

if Path(ADJ_FILLED_CSV).is_file():
    adjf = read_csv_robust(ADJ_FILLED_CSV)
    # Fill Gold_* if reviewer marked Y but left Gold blank
    def _fill_gold(row, model_col, gold_col, yn_col):
        v = str(row.get(gold_col, "")).strip()
        if v: return v
        ok = row.get(yn_col, "")
        ok = str(ok).strip().upper()
        if ok in {"Y","YES","TRUE","1"}:
            return row.get(model_col, "")
        return ""

    adjf["Gold_Major"] = adjf.apply(lambda r: _fill_gold(r, "Model_Pred_Major", "Gold_Major", "Human_Major_Correct_YN"), axis=1)
    adjf["Gold_Minor"] = adjf.apply(lambda r: _fill_gold(r, "Model_Pred_Minor", "Gold_Minor", "Human_Minor_Correct_YN"), axis=1)

    # Bridge to practical‑eval schema
    preds_eval = pd.DataFrame({
        COL_RECORD_ID:  adjf.get("Record_ID", pd.Series(range(1, len(adjf)+1))),
        COL_TRUE_MAJOR: adjf["Gold_Major"].astype(str).str.strip().replace({"": np.nan}).fillna("Other"),
        COL_PRED_MAJOR: adjf.get("Model_Pred_Major", pd.Series(["Other"]*len(adjf))).astype(str).str.strip().replace({"": "Other"}),
        COL_TRUE_MINOR: adjf["Gold_Minor"].astype(str).str.strip().replace({"": np.nan}).fillna("I"),
        COL_PRED_MINOR: adjf.get("Model_Pred_Minor", pd.Series(["I"]*len(adjf))).astype(str).str.strip().replace({"": "I"}),
    })

    # --- Practical eval helpers ---
    def confusion(df: pd.DataFrame, true_col: str, pred_col: str, labels: list) -> pd.DataFrame:
        cm = pd.crosstab(df[true_col], df[pred_col], rownames=["True"], colnames=["Pred"], dropna=False)
        return cm.reindex(index=labels, columns=labels, fill_value=0)

    def accuracy_and_error_rate(df: pd.DataFrame, true_col: str, pred_col: str):
        total = len(df); correct = (df[true_col] == df[pred_col]).sum()
        acc = correct/total if total else float('nan')
        return acc, (1.0-acc if total else float('nan')), correct, total

    def minor_delta_severity(true_minor, pred_minor):
        ti = MINOR_IDX.get(str(true_minor), None); pi = MINOR_IDX.get(str(pred_minor), None)
        if ti is None or pi is None: return 0 if ti == pi else 2
        d = abs(ti - pi); return (0,2,3,4)[d] if d <= 3 else 4

    def severity_score(true_major, true_minor, pred_major, pred_minor):
        return minor_delta_severity(true_minor, pred_minor) if true_major == pred_major else SEVERITY_MAJOR_GRID.get((true_major, pred_major), 0)

    def comp_value(major, minor):
        base = COST_MAP_MAJOR.get(major, COST_MAP_MAJOR.get("Other", 0.0))
        step = COST_MAP_MINOR.get(str(minor), COST_MAP_MINOR["I"]) if str(minor) in COST_MAP_MINOR else COST_MAP_MINOR["I"]
        return float(base + step)

    def sev_multiplier(score):
        return SEVERITY_COST_MULTIPLIER.get(int(score), 1.0)

    def add_row_metrics(df):
        df = df.copy()
        df["severity_score"] = df.apply(lambda r: severity_score(r[COL_TRUE_MAJOR], r[COL_TRUE_MINOR], r[COL_PRED_MAJOR], r[COL_PRED_MINOR]), axis=1)
        df["true_comp"] = df.apply(lambda r: comp_value(r[COL_TRUE_MAJOR], r[COL_TRUE_MINOR]), axis=1)
        df["pred_comp"] = df.apply(lambda r: comp_value(r[COL_PRED_MAJOR], r[COL_PRED_MINOR]), axis=1)
        df["cost_base"] = (df["true_comp"] - df["pred_comp"]).abs()
        df["cost_weighted"] = np.where(APPLY_SEVERITY_MULTIPLIER, df["cost_base"] * df["severity_score"].map(sev_multiplier), df["cost_base"])
        return df

    def pivot_sum(df, true_col, pred_col, value_col, row_labels, col_labels):
        out = pd.DataFrame(0.0, index=row_labels, columns=col_labels)
        grp = df.groupby([true_col, pred_col], dropna=False)[value_col].sum()
        for (r,c), v in grp.items():
            if r in out.index and c in out.columns:
                out.loc[r, c] = float(v)
        return out

    def label28(major, minor):
        m = major if major in MAJOR_CLASSES else "Other"
        s = str(minor) if str(minor) in MINOR_CLASSES else "I"
        return f"{m} {s}"

    def summarize_costs(df: pd.DataFrame, true_col: str, pred_col: str, cost_map: dict):
        mask_wrong = df[true_col] != df[pred_col]
        if mask_wrong.any():
            deltas = (df.loc[mask_wrong, true_col].map(cost_map).fillna(0.0) - df.loc[mask_wrong, pred_col].map(cost_map).fillna(0.0)).abs()
            return {"n_errors": int(mask_wrong.sum()), "total_cost": float(deltas.sum()), "avg_cost_per_error": float(deltas.mean())}
        return {"n_errors": 0, "total_cost": 0.0, "avg_cost_per_error": 0.0}

    def export_csv(df: pd.DataFrame, name: str):
        out = Path(OUTPUT_DIR)/name; df.to_csv(out, index=True); return str(out)

    def export_json(obj, name: str):
        out = Path(OUTPUT_DIR)/name; json.dump(obj, open(out, "w", encoding="utf-8"), indent=2); return str(out)

    # Normalize strings
    for c in [COL_TRUE_MAJOR, COL_TRUE_MINOR, COL_PRED_MAJOR, COL_PRED_MINOR]:
        preds_eval[c] = preds_eval[c].astype(str).str.strip()

    # Major evaluation
    major_acc, major_err, major_correct, major_total = accuracy_and_error_rate(preds_eval, COL_TRUE_MAJOR, COL_PRED_MAJOR)
    major_gate_pass = (major_err <= ACCEPTABLE_MAJOR_ERROR_RATE)
    cm_major = confusion(preds_eval, COL_TRUE_MAJOR, COL_PRED_MAJOR, MAJOR_CLASSES)
    export_csv(cm_major, "confusion_major_counts.csv")
    preds_enriched = add_row_metrics(preds_eval)
    cm_major_sevsum  = pivot_sum(preds_enriched, COL_TRUE_MAJOR, COL_PRED_MAJOR, "severity_score", MAJOR_CLASSES, MAJOR_CLASSES)
    cm_major_sevavg  = cm_major_sevsum / cm_major.replace(0, np.nan)
    cm_major_costsum = pivot_sum(preds_enriched, COL_TRUE_MAJOR, COL_PRED_MAJOR, "cost_base", MAJOR_CLASSES, MAJOR_CLASSES)
    cm_major_costavg = cm_major_costsum / cm_major.replace(0, np.nan)
    cm_major_costsum_w = pivot_sum(preds_enriched, COL_TRUE_MAJOR, COL_PRED_MAJOR, "cost_weighted", MAJOR_CLASSES, MAJOR_CLASSES)
    cm_major_costavg_w = cm_major_costsum_w / cm_major.replace(0, np.nan)
    export_csv(cm_major_sevsum,  "confusion_major_severity_sum.csv")
    export_csv(cm_major_sevavg,  "confusion_major_severity_avg.csv")
    export_csv(cm_major_costsum, "confusion_major_cost_sum.csv")
    export_csv(cm_major_costavg, "confusion_major_cost_avg.csv")
    export_csv(cm_major_costsum_w, "confusion_major_cost_sum_weighted.csv")
    export_csv(cm_major_costavg_w, "confusion_major_cost_avg_weighted.csv")

    # 28×28
    preds_enriched["true_28"] = preds_enriched.apply(lambda r: label28(r[COL_TRUE_MAJOR], r[COL_TRUE_MINOR]), axis=1)
    preds_enriched["pred_28"] = preds_enriched.apply(lambda r: label28(r[COL_PRED_MAJOR], r[COL_PRED_MINOR]), axis=1)
    labels28 = [f"{M} {m}" for M in MAJOR_CLASSES for m in MINOR_CLASSES]
    cm_28 = confusion(preds_enriched.rename(columns={"true_28":"__t28","pred_28":"__p28"}), "__t28", "__p28", labels28)
    cm_28_sevsum  = pivot_sum(preds_enriched.rename(columns={"true_28":"__t28","pred_28":"__p28"}), "__t28","__p28","severity_score", labels28, labels28)
    cm_28_sevavg  = cm_28_sevsum / cm_28.replace(0, np.nan)
    cm_28_costsum = pivot_sum(preds_enriched.rename(columns={"true_28":"__t28","pred_28":"__p28"}), "__t28","__p28","cost_base", labels28, labels28)
    cm_28_costavg = cm_28_costsum / cm_28.replace(0, np.nan)
    cm_28_costsum_w = pivot_sum(preds_enriched.rename(columns={"true_28":"__t28","pred_28":"__p28"}), "__t28","__p28","cost_weighted", labels28, labels28)
    cm_28_costavg_w = cm_28_costsum_w / cm_28.replace(0, np.nan)
    export_csv(cm_28,             "confusion_major_minor_counts_28x28.csv")
    export_csv(cm_28_sevsum,      "confusion_major_minor_severity_sum_28x28.csv")
    export_csv(cm_28_sevavg,      "confusion_major_minor_severity_avg_28x28.csv")
    export_csv(cm_28_costsum,     "confusion_major_minor_cost_sum_28x28.csv")
    export_csv(cm_28_costavg,     "confusion_major_minor_cost_avg_28x28.csv")
    export_csv(cm_28_costsum_w,   "confusion_major_minor_cost_sum_weighted_28x28.csv")
    export_csv(cm_28_costavg_w,   "confusion_major_minor_cost_avg_weighted_28x28.csv")

    # Minor (flat)
    minor_acc, minor_err, minor_correct, minor_total = accuracy_and_error_rate(preds_eval, COL_TRUE_MINOR, COL_PRED_MINOR)
    minor_gate_pass = (minor_err <= ACCEPTABLE_MINOR_ERROR_RATE)
    cm_minor = confusion(preds_eval, COL_TRUE_MINOR, COL_PRED_MINOR, MINOR_CLASSES)
    export_csv(cm_minor, "confusion_minor.csv")
    # Minor‑only severity summaries
    preds_minor = preds_enriched.copy()
    preds_minor["minor_severity_only"] = preds_minor.apply(lambda r: minor_delta_severity(r[COL_TRUE_MINOR], r[COL_PRED_MINOR]), axis=1)
    cm_minor_sevsum = pivot_sum(preds_minor, COL_TRUE_MINOR, COL_PRED_MINOR, "minor_severity_only", MINOR_CLASSES, MINOR_CLASSES)
    cm_minor_sevavg = cm_minor_sevsum / cm_minor.replace(0, np.nan)
    export_csv(cm_minor_sevsum, "confusion_minor_severity_sum.csv")
    export_csv(cm_minor_sevavg, "confusion_minor_severity_avg.csv")

    # Cost summaries
    def cost_of_error_matrix(df: pd.DataFrame, true_col: str, pred_col: str, cost_map: dict, labels: list) -> pd.DataFrame:
        pairs = []
        for _, row in df.iterrows():
            t, p = row[true_col], row[pred_col]
            t_cost = cost_map.get(t, 0.0); p_cost = cost_map.get(p, 0.0)
            delta = abs(t_cost - p_cost) if (t != p) else 0.0
            pairs.append((t,p,delta))
        cost_df = pd.DataFrame(pairs, columns=["True","Pred","CostDelta"])
        piv = cost_df.pivot_table(index="True", columns="Pred", values="CostDelta", aggfunc="sum", fill_value=0.0)
        return piv.reindex(index=labels, columns=labels, fill_value=0.0)

    export_csv(cost_of_error_matrix(preds_eval, COL_TRUE_MAJOR, COL_PRED_MAJOR, COST_MAP_MAJOR, MAJOR_CLASSES), "cost_matrix_major.csv")
    export_csv(cost_of_error_matrix(preds_eval, COL_TRUE_MINOR, COL_PRED_MINOR, COST_MAP_MINOR, MINOR_CLASSES), "cost_matrix_minor.csv")

    # Metrics JSON
    def summarize_costs_simple(df, true_col, pred_col, cost_map):
        mask_wrong = df[true_col] != df[pred_col]
        if mask_wrong.any():
            deltas = (df.loc[mask_wrong, true_col].map(cost_map).fillna(0.0) - df.loc[mask_wrong, pred_col].map(cost_map).fillna(0.0)).abs()
            total_cost = float(deltas.sum()); avg_cost = float(deltas.mean()); n_err = int(mask_wrong.sum())
        else:
            total_cost = avg_cost = 0.0; n_err = 0
        return dict(n_errors=n_err, total_cost=total_cost, avg_cost_per_error=avg_cost)

    metrics_major = {
        "accuracy": float(major_acc), "error_rate": float(major_err),
        "correct": int(major_correct), "total": int(major_total),
        "accept_error_threshold": ACCEPTABLE_MAJOR_ERROR_RATE,
        "pass_gate": bool(major_gate_pass),
        "cost_summary": summarize_costs_simple(preds_eval, COL_TRUE_MAJOR, COL_PRED_MAJOR, COST_MAP_MAJOR),
    }
    metrics_minor = {
        "accuracy": float(minor_acc), "error_rate": float(minor_err),
        "correct": int(minor_correct), "total": int(minor_total),
        "accept_error_threshold": ACCEPTABLE_MINOR_ERROR_RATE,
        "pass_gate": bool(minor_gate_pass),
        "cost_summary": summarize_costs_simple(preds_eval, COL_TRUE_MINOR, COL_PRED_MINOR, COST_MAP_MINOR),
    }
    export_json(metrics_major, "metrics_major.json"); export_json(metrics_minor, "metrics_minor.json")

    # KPI: Severity‑Weighted Cost Index (SWCI)
    mask_mis = preds_enriched["severity_score"] > 0
    total_weighted_cost = float(preds_enriched.loc[mask_mis, "cost_weighted"].sum())
    total_payroll = float(preds_enriched["true_comp"].sum())
    n_records = int(len(preds_enriched))
    mis_rate = float(mask_mis.mean())
    SWCI = (1_000_000.0 * total_weighted_cost) / max(total_payroll, 1.0)
    SWCI_per_1k = (1_000.0 * total_weighted_cost) / max(n_records, 1)

    # Merge SUPPORT (if present)
    support = None
    try:
        with open(Path(OUTPUT_DIR)/"support_metrics.json", "r", encoding="utf-8") as f:
            support = json.load(f)
    except Exception:
        support = None

    summary = {
        "primary": {
            "major": metrics_major,
            "minor": metrics_minor,
            "gates": {"major_pass": metrics_major["pass_gate"], "minor_pass": metrics_minor["pass_gate"]},
            "kpi": {"SWCI_per_$1M": SWCI, "SWCI_per_1000_predictions": SWCI_per_1k},
        },
        "support": support or {},
    }
    export_json(summary, "summary_overview.json")

    # Review queues: base cost + severity‑weighted
    def _error_rows_with_cost(df, true_col, pred_col, cost_map, tag):
        mask = df[true_col] != df[pred_col]
        if not mask.any():
            return pd.DataFrame(columns=["record_id","true","pred","cost_delta","dimension"])
        t_cost = df.loc[mask, true_col].map(cost_map).fillna(0.0)
        p_cost = df.loc[mask, pred_col].map(cost_map).fillna(0.0)
        d = (t_cost - p_cost).abs()
        return pd.DataFrame({
            "record_id": df.loc[mask, COL_RECORD_ID],
            "true": df.loc[mask, true_col],
            "pred": df.loc[mask, pred_col],
            "cost_delta": d,
            "dimension": tag,
        })

    err_major = _error_rows_with_cost(preds_eval, COL_TRUE_MAJOR, COL_PRED_MAJOR, COST_MAP_MAJOR, "major")
    err_minor = _error_rows_with_cost(preds_eval, COL_TRUE_MINOR, COL_PRED_MINOR, COST_MAP_MINOR, "minor")
    review = pd.concat([err_major, err_minor], ignore_index=True)
    review = review.sort_values("cost_delta", ascending=False).head(200).reset_index(drop=True)
    review_path = Path(OUTPUT_DIR)/"errors_sampled_for_review.csv"; review.to_csv(review_path, index=False)

    review_sw = review.merge(
        preds_enriched[[COL_RECORD_ID, "cost_weighted"]],
        left_on="record_id", right_on=COL_RECORD_ID, how="left"
    ).rename(columns={"cost_weighted":"severity_weighted_cost"})
    review_sw = review_sw.sort_values(["severity_weighted_cost","cost_delta"], ascending=False).head(200)
    review_sw_path = Path(OUTPUT_DIR)/"errors_sampled_for_review_severity_weighted.csv"; review_sw.to_csv(review_sw_path, index=False)

    print("Practical eval artifacts written to:", OUTPUT_DIR)

# ==========================
# Section 10 — Export Recap
# ==========================
print("Artifacts in", OUTPUT_DIR)
for p in sorted(Path(OUTPUT_DIR).glob("*.csv"))[:50]:
    print(" -", p.name)
for p in sorted(Path(OUTPUT_DIR).glob("*.json"))[:50]:
    print(" -", p.name)

print("\nDone. If you changed the prompt, bump PROMPT_VERSION and re‑run Sections 6–10.")
