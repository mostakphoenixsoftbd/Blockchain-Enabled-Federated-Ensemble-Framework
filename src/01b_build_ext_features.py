"""Extended first-24h features from the ICU module and from hosp prescriptions/microbiology.

usage: python 01b_build_ext_features.py [--chart]

Writes data/ext_icu_rx_micro.parquet (one row per cohort hadm_id) and, with --chart,
data/ext_chart.parquet (bedside vital signs and GCS from icu/chartevents).
Both are joined onto data/cohort.parquet by hadm_id without changing its rows, so the
hospital partition, the patient-level splits and the test patients stay identical to the
primary analysis (paired comparison).

Leakage rules (same window as the laboratory features): only events that start in
[admittime - 6 h, admittime + 24 h) are used. Infusion volumes are prorated to the part of
the infusion inside the window. Microbiology uses culture *orders* (charttime), never
results. Comfort-care medications are not used.
"""
import argparse
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
import os
MIMIC = os.environ.get("MIMIC_DIR", "E:/physionet/mimiciv/3.1")  # set MIMIC_DIR to your MIMIC-IV v3.1 folder
HOSP, ICU = f"{MIMIC}/hosp", f"{MIMIC}/icu"

ap = argparse.ArgumentParser()
ap.add_argument("--chart", action="store_true", help="also build vital signs from icu/chartevents")
args = ap.parse_args()

tmp = DATA / "duckdb_tmp"; tmp.mkdir(exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("SET memory_limit='3GB'")
con.execute(f"SET temp_directory='{tmp.as_posix()}'")
con.execute("SET preserve_insertion_order=false")

con.execute(f"""
CREATE TABLE coh AS
SELECT hadm_id, subject_id, admittime,
       admittime - INTERVAL 6 HOUR AS w0, admittime + INTERVAL 24 HOUR AS w1
FROM read_parquet('{(DATA / 'cohort.parquet').as_posix()}')
""")


def step(name, sql):
    t = time.time()
    con.execute(sql)
    n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
    print(f"{name}: {n} admissions with data ({time.time() - t:.0f}s)", flush=True)


VASO = (221906, 221289, 222315, 221749, 221662, 221653)  # norepi, epi, vasopressin, phenylephrine, dopamine, dobutamine
SEDATION = (222168, 221668)                                # propofol, midazolam
CRYSTALLOID = (225158, 225828)                             # NaCl 0.9 %, lactated Ringer's
DIALYSIS = (225441, 225802, 225803, 225805, 225809, 225955)
URINE = (226557, 226558, 226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584, 227489)
GU_IRRIGANT_IN = 227488

# ---------------------------------------------------------------- ICU stays
step("icu", f"""
CREATE TABLE icu AS
SELECT c.hadm_id,
       sum(date_diff('minute', greatest(s.intime, c.w0), least(s.outtime, c.w1))) / 60.0 AS icu_window_hrs,
       arg_min(s.first_careunit, s.intime) AS unit
FROM coh c JOIN read_csv_auto('{ICU}/icustays.csv.gz') s USING (hadm_id)
WHERE s.intime < c.w1 AND s.outtime > c.w0
GROUP BY c.hadm_id
""")

step("inp", f"""
CREATE TABLE inp AS
SELECT c.hadm_id,
       count(DISTINCT e.itemid) FILTER (WHERE e.itemid IN {VASO}) AS icu_vasoactive_cnt,
       max((e.itemid IN {SEDATION})::INT) AS icu_sedation,
       sum(CASE WHEN e.itemid IN {CRYSTALLOID} AND e.amountuom IN ('mL', 'L') THEN
           e.amount * (CASE WHEN e.amountuom = 'L' THEN 1000 ELSE 1 END) * least(1.0, greatest(0.0,
               date_diff('second', greatest(e.starttime, c.w0), least(e.endtime, c.w1))
               / greatest(date_diff('second', e.starttime, e.endtime), 1)))
           END) AS icu_crystalloid_ml,
       median(e.patientweight) FILTER (WHERE e.patientweight BETWEEN 20 AND 300) AS icu_weight_kg
FROM coh c JOIN read_csv('{ICU}/inputevents.csv.gz', header=true, auto_detect=true) e USING (hadm_id)
WHERE e.starttime < c.w1 AND e.endtime > c.w0
GROUP BY c.hadm_id
""")

step("proc", f"""
CREATE TABLE proc AS
SELECT c.hadm_id,
       max((e.itemid = 225792)::INT) AS icu_invasive_vent,
       max((e.itemid = 225794)::INT) AS icu_noninvasive_vent,
       max((e.itemid IN {DIALYSIS})::INT) AS icu_dialysis
FROM coh c JOIN read_csv_auto('{ICU}/procedureevents.csv.gz') e USING (hadm_id)
WHERE e.starttime < c.w1 AND e.endtime > c.w0
GROUP BY c.hadm_id
""")

step("uo", f"""
CREATE TABLE uo AS
SELECT c.hadm_id,
       greatest(sum(CASE WHEN e.itemid = {GU_IRRIGANT_IN} THEN -e.value ELSE e.value END), 0) AS icu_urine_ml
FROM coh c JOIN read_csv_auto('{ICU}/outputevents.csv.gz') e USING (hadm_id)
WHERE e.itemid IN {URINE + (GU_IRRIGANT_IN,)} AND e.charttime >= c.w0 AND e.charttime < c.w1
GROUP BY c.hadm_id
""")

# ---------------------------------------------------------------- prescriptions (orders started in the window)
ANTIMICROBIAL = ("vancomycin|piperacillin|meropenem|cefepime|ceftriaxone|ceftazidime|cefazolin|levofloxacin|"
                 "ciprofloxacin|metronidazole|azithromycin|ampicillin|nafcillin|aztreonam|linezolid|daptomycin|"
                 "gentamicin|tobramycin|clindamycin|ertapenem|imipenem|cefuroxime|cefoxitin|micafungin|"
                 "fluconazole|acyclovir|sulfamethoxazole|doxycycline|ceftaroline")
step("rx", f"""
CREATE TABLE rx AS
WITH r AS (
  SELECT c.hadm_id, lower(p.drug) AS drug, upper(coalesce(p.route, '')) AS route
  FROM coh c JOIN read_csv('{HOSP}/prescriptions.csv.gz', header=true, auto_detect=true,
                           types={{'gsn':'VARCHAR','ndc':'VARCHAR','dose_val_rx':'VARCHAR',
                                   'form_val_disp':'VARCHAR','poe_seq':'VARCHAR'}}) p USING (hadm_id)
  WHERE p.starttime >= c.w0 AND p.starttime < c.w1
)
SELECT hadm_id,
       count(DISTINCT drug) AS rx_distinct_cnt,
       count(DISTINCT regexp_extract(drug, '({ANTIMICROBIAL})')) FILTER (WHERE regexp_matches(drug, '{ANTIMICROBIAL}')) AS rx_antimicrobial_cnt,
       max((regexp_matches(drug, '{ANTIMICROBIAL}') AND route LIKE 'IV%')::INT) AS rx_iv_antimicrobial,
       max(regexp_matches(drug, 'norepinephrine|vasopressin|phenylephrine|dopamine|epinephrine')::INT) AS rx_vasopressor,
       max(regexp_matches(drug, 'insulin')::INT) AS rx_insulin,
       max(regexp_matches(drug, 'furosemide|bumetanide|torsemide')::INT) AS rx_loop_diuretic,
       max(regexp_matches(drug, 'hydrocortisone|methylprednisolone|dexamethasone|prednisone')::INT) AS rx_corticosteroid,
       max((regexp_matches(drug, 'heparin') AND NOT regexp_matches(drug, 'flush|lock')
            AND route LIKE 'IV%')::INT) AS rx_iv_heparin
FROM r GROUP BY hadm_id
""")

# ---------------------------------------------------------------- microbiology orders (many rows lack hadm_id: match on patient + time)
step("micro", f"""
CREATE TABLE micro AS
SELECT c.hadm_id,
       count(DISTINCT m.micro_specimen_id) AS micro_specimen_cnt,
       max((m.spec_type_desc = 'BLOOD CULTURE')::INT) AS micro_blood_culture,
       max((m.spec_type_desc = 'URINE')::INT) AS micro_urine_culture,
       max((m.spec_type_desc IN ('SPUTUM', 'BRONCHOALVEOLAR LAVAGE', 'BRONCHIAL WASHINGS',
                                 'Mini-BAL'))::INT) AS micro_respiratory_culture
FROM coh c JOIN read_csv('{HOSP}/microbiologyevents.csv.gz', header=true, auto_detect=true,
                         types={{'dilution_value':'DOUBLE','quantity':'VARCHAR','comments':'VARCHAR'}}) m
     ON m.subject_id = c.subject_id
WHERE m.charttime >= c.w0 AND m.charttime < c.w1
GROUP BY c.hadm_id
""")

out = DATA / "ext_icu_rx_micro.parquet"
con.execute(f"""
COPY (
SELECT c.hadm_id,
       (icu.hadm_id IS NOT NULL)::INT AS icu_in_window,
       coalesce(icu.icu_window_hrs, 0) AS icu_window_hrs,
       coalesce(icu.unit IN ('Medical Intensive Care Unit (MICU)', 'Medical/Surgical Intensive Care Unit (MICU/SICU)'), false)::INT AS icu_unit_medical,
       coalesce(icu.unit IN ('Surgical Intensive Care Unit (SICU)', 'Trauma SICU (TSICU)'), false)::INT AS icu_unit_surgical,
       coalesce(icu.unit IN ('Cardiac Vascular Intensive Care Unit (CVICU)', 'Coronary Care Unit (CCU)'), false)::INT AS icu_unit_cardiac,
       coalesce(icu.unit LIKE 'Neuro%', false)::INT AS icu_unit_neuro,
       coalesce(inp.icu_vasoactive_cnt, 0) AS icu_vasoactive_cnt,
       coalesce(inp.icu_sedation, 0) AS icu_sedation,
       coalesce(inp.icu_crystalloid_ml, 0) AS icu_crystalloid_ml,
       inp.icu_weight_kg,
       coalesce(proc.icu_invasive_vent, 0) AS icu_invasive_vent,
       coalesce(proc.icu_noninvasive_vent, 0) AS icu_noninvasive_vent,
       coalesce(proc.icu_dialysis, 0) AS icu_dialysis,
       CASE WHEN icu.hadm_id IS NOT NULL THEN uo.icu_urine_ml END AS icu_urine_ml,
       CASE WHEN icu.icu_window_hrs >= 4 THEN uo.icu_urine_ml / icu.icu_window_hrs END AS icu_urine_rate,
       coalesce(rx.rx_distinct_cnt, 0) AS rx_distinct_cnt,
       coalesce(rx.rx_antimicrobial_cnt, 0) AS rx_antimicrobial_cnt,
       {", ".join(f"coalesce(rx.{k}, 0) AS {k}" for k in ("rx_iv_antimicrobial", "rx_vasopressor", "rx_insulin",
                                                         "rx_loop_diuretic", "rx_corticosteroid", "rx_iv_heparin"))},
       coalesce(micro.micro_specimen_cnt, 0) AS micro_specimen_cnt,
       {", ".join(f"coalesce(micro.{k}, 0) AS {k}" for k in ("micro_blood_culture", "micro_urine_culture",
                                                            "micro_respiratory_culture"))}
FROM coh c
LEFT JOIN icu USING (hadm_id) LEFT JOIN inp USING (hadm_id) LEFT JOIN proc USING (hadm_id)
LEFT JOIN uo USING (hadm_id) LEFT JOIN rx USING (hadm_id) LEFT JOIN micro USING (hadm_id)
) TO '{out.as_posix()}' (FORMAT parquet)
""")
print(f"wrote {out.name}")

# ---------------------------------------------------------------- bedside vital signs (icu/chartevents)
if args.chart:
    VITALS = {  # name: (itemids, plausible range)
        "hr": ((220045,), (0, 300)),
        "sbp": ((220179, 220050), (20, 300)),
        "map": ((220181, 220052), (10, 250)),
        "rr": ((220210, 224690), (0, 80)),
        "spo2": ((220277,), (10, 100)),
        "temp": ((223762, 223761), (25, 45)),
    }
    GCS = {"gcs_eye": 220739, "gcs_verbal": 223900, "gcs_motor": 223901}
    ids = sorted({i for v in VITALS.values() for i in v[0]} | set(GCS.values()))
    # pass 1 streams the 3.3 GB file to a small parquet (item filter only, no join) so it
    # fits in ~1 GB of RAM on a shared machine; pass 2 applies the time window
    con.execute("PRAGMA threads=2")
    con.execute("SET memory_limit='1GB'")
    ce_raw = tmp / "chartevents_vitals.parquet"
    t = time.time()
    if not ce_raw.exists():
        con.execute(f"""
        COPY (SELECT hadm_id, charttime, itemid, valuenum
              FROM read_csv('{ICU}/chartevents.csv.gz', header=true, auto_detect=false,
                   columns={{'subject_id':'BIGINT','hadm_id':'BIGINT','stay_id':'BIGINT','caregiver_id':'BIGINT',
                            'charttime':'TIMESTAMP','storetime':'TIMESTAMP','itemid':'INTEGER','value':'VARCHAR',
                            'valuenum':'DOUBLE','valueuom':'VARCHAR','warning':'INTEGER'}})
              WHERE itemid IN ({",".join(map(str, ids))}) AND valuenum IS NOT NULL AND hadm_id IS NOT NULL)
        TO '{ce_raw.as_posix()}.part' (FORMAT parquet)
        """)
        Path(f"{ce_raw}.part").replace(ce_raw)  # a crash mid-scan leaves no half file behind
    print(f"chartevents filtered in {time.time() - t:.0f}s", flush=True)
    con.execute(f"""
    CREATE TABLE ce AS
    SELECT c.hadm_id, e.itemid,
           CASE WHEN e.itemid = 223761 THEN (e.valuenum - 32) * 5 / 9 ELSE e.valuenum END AS v
    FROM coh c JOIN read_parquet('{ce_raw.as_posix()}') e USING (hadm_id)
    WHERE e.charttime >= c.w0 AND e.charttime < c.w1
    """)
    aggs = []
    for name, (items, (lo, hi)) in VITALS.items():
        cond = f"itemid IN ({','.join(map(str, items))}) AND v BETWEEN {lo} AND {hi}"
        # suffixes _lo/_hi/_avg keep these out of the Hospital C lab-missingness simulation
        aggs += [f"min(v) FILTER (WHERE {cond}) AS vs_{name}_lo",
                 f"max(v) FILTER (WHERE {cond}) AS vs_{name}_hi",
                 f"avg(v) FILTER (WHERE {cond}) AS vs_{name}_avg"]
    aggs += [f"min(v) FILTER (WHERE itemid = {i}) AS vs_{k}_lo" for k, i in GCS.items()]
    out2 = DATA / "ext_chart.parquet"
    con.execute(f"""
    COPY (SELECT c.hadm_id, a.* EXCLUDE (hadm_id)
          FROM coh c LEFT JOIN (SELECT hadm_id, {", ".join(aggs)} FROM ce GROUP BY hadm_id) a USING (hadm_id))
    TO '{out2.as_posix()}' (FORMAT parquet)
    """)
    print(f"wrote {out2.name}")
