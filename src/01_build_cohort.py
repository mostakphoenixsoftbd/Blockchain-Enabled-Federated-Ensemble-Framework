"""Build the in-hospital mortality cohort from MIMIC-IV v3.1 (hosp module).

Unit of analysis: one hospital admission (hadm_id).
Inclusion: age >= 18 at admission, length of stay >= 24 h, alive at 24 h.
Prediction time: 24 h after admission. Features use only information available
by then (demographics, admission context, first-24h labs, history from
*previous* admissions). Discharge diagnoses of the index admission are never used.
Outcome: hospital_expire_flag.
"""
import time
from pathlib import Path

import duckdb

import os
HOSP = os.environ.get("MIMIC_DIR", "E:/physionet/mimiciv/3.1") + "/hosp"  # set MIMIC_DIR to your MIMIC-IV v3.1 folder
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

LABS = {
    50912: "creatinine", 51006: "bun", 50983: "sodium", 50971: "potassium",
    50902: "chloride", 50882: "bicarbonate", 50868: "anion_gap", 50931: "glucose",
    50893: "calcium", 50960: "magnesium", 50970: "phosphate", 51222: "hemoglobin",
    51221: "hematocrit", 51301: "wbc", 51265: "platelets", 51277: "rdw",
    50813: "lactate", 50820: "ph", 50885: "bilirubin", 50861: "alt", 50878: "ast",
    50863: "alp", 50862: "albumin", 51237: "inr", 51274: "pt", 51275: "ptt",
    51244: "lymphocytes_pct", 51256: "neutrophils_pct", 51003: "troponin_t",
}

con = duckdb.connect()
con.execute("PRAGMA threads=8")

lab_cache = DATA / "labs_filtered.parquet"
if not lab_cache.exists():
    t = time.time()
    ids = ",".join(map(str, LABS))
    con.execute(f"""
        COPY (
          SELECT subject_id, hadm_id, itemid, charttime, valuenum
          FROM read_csv('{HOSP}/labevents.csv.gz', header=true, quote='"', escape='"', delim=',',
               columns={{'labevent_id':'BIGINT','subject_id':'BIGINT','hadm_id':'BIGINT',
                        'specimen_id':'BIGINT','itemid':'INTEGER','order_provider_id':'VARCHAR',
                        'charttime':'TIMESTAMP','storetime':'TIMESTAMP','value':'VARCHAR',
                        'valuenum':'DOUBLE','valueuom':'VARCHAR','ref_range_lower':'DOUBLE',
                        'ref_range_upper':'DOUBLE','flag':'VARCHAR','priority':'VARCHAR',
                        'comments':'VARCHAR'}})
          WHERE itemid IN ({ids}) AND valuenum IS NOT NULL
        ) TO '{lab_cache.as_posix()}' (FORMAT parquet)
    """)
    print(f"labs filtered in {time.time()-t:.0f}s")

con.execute(f"""
CREATE TABLE adm AS
SELECT a.*, p.gender, p.anchor_age, p.anchor_year, p.anchor_year_group,
       p.anchor_age + (year(a.admittime) - p.anchor_year) AS age
FROM read_csv_auto('{HOSP}/admissions.csv.gz') a
JOIN read_csv_auto('{HOSP}/patients.csv.gz') p USING (subject_id)
""")

# history from previous admissions (all admissions, before index admittime)
con.execute(f"""
CREATE TABLE dx AS
SELECT d.subject_id, d.hadm_id, d.icd_code, d.icd_version, a.admittime
FROM read_csv_auto('{HOSP}/diagnoses_icd.csv.gz', types={{'icd_code':'VARCHAR'}}) d
JOIN adm a USING (hadm_id)
""")

# Charlson-style comorbidity groups (ICD-9 and ICD-10 prefixes)
CHARLSON = {
    "chf": (["428"], ["I50"]),
    "mi": (["410", "412"], ["I21", "I22", "I252"]),
    "pvd": (["4439", "441", "7854"], ["I70", "I71", "I739"]),
    "cvd": (["430", "431", "432", "433", "434", "435", "436", "437", "438"], ["I6", "G45"]),
    "dementia": (["290"], ["F00", "F01", "F02", "F03", "G30"]),
    "copd": (["490", "491", "492", "493", "494", "495", "496"], ["J40", "J41", "J42", "J43", "J44", "J45", "J47"]),
    "diabetes": (["250"], ["E10", "E11", "E13"]),
    "ckd": (["585", "586"], ["N18", "N19"]),
    "liver": (["571", "5722", "5723", "5724"], ["K70", "K72", "K74", "K766"]),
    "cancer": ([str(i) for i in range(140, 173)] + [str(i) for i in range(174, 196)],
               ["C0", "C1", "C2", "C3", "C40", "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C5", "C6", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C81", "C82", "C83", "C84", "C85", "C88", "C9"]),
    "metastatic": (["196", "197", "198", "199"], ["C77", "C78", "C79", "C80"]),
    "hiv": (["042"], ["B20"]),
}
cases = []
for k, (i9, i10) in CHARLSON.items():
    c9 = " OR ".join(f"starts_with(h.icd_code,'{p}')" for p in i9)
    c10 = " OR ".join(f"starts_with(h.icd_code,'{p}')" for p in i10)
    cases.append(f"max(CASE WHEN (h.icd_version=9 AND ({c9})) OR (h.icd_version=10 AND ({c10})) THEN 1 ELSE 0 END) AS hx_{k}")

con.execute(f"""
CREATE TABLE hx AS
SELECT i.hadm_id, {", ".join(cases)}
FROM adm i JOIN dx h ON h.subject_id=i.subject_id AND h.admittime < i.admittime
GROUP BY i.hadm_id
""")

con.execute("""
CREATE TABLE prior AS
SELECT i.hadm_id,
       count(p.hadm_id) AS n_prior_adm,
       date_diff('day', max(p.dischtime), i.admittime) AS days_since_last_disch
FROM adm i LEFT JOIN adm p ON p.subject_id=i.subject_id AND p.dischtime <= i.admittime
GROUP BY i.hadm_id, i.admittime
""")

lab_aggs = []
for item, name in LABS.items():
    lab_aggs += [
        f"min(valuenum) FILTER (WHERE itemid={item}) AS {name}_min",
        f"max(valuenum) FILTER (WHERE itemid={item}) AS {name}_max",
        f"avg(valuenum) FILTER (WHERE itemid={item}) AS {name}_mean",
    ]
con.execute(f"""
CREATE TABLE labs AS
SELECT l.hadm_id, count(*) AS n_lab_results, {", ".join(lab_aggs)}
FROM read_parquet('{lab_cache.as_posix()}') l
JOIN adm a USING (hadm_id)
WHERE l.charttime >= a.admittime - INTERVAL 6 HOUR
  AND l.charttime <  a.admittime + INTERVAL 24 HOUR
GROUP BY l.hadm_id
""")

con.execute(f"""
COPY (
SELECT a.subject_id, a.hadm_id, a.admittime,
       a.hospital_expire_flag AS label,
       a.age, a.gender, a.admission_type, a.admission_location, a.insurance,
       a.marital_status, a.race, a.language, a.anchor_year_group,
       (a.edregtime IS NOT NULL)::INT AS came_via_ed,
       hour(a.admittime) AS admit_hour,
       (dayofweek(a.admittime) IN (0,6))::INT AS weekend_admit,
       pr.n_prior_adm, pr.days_since_last_disch,
       {", ".join(f"coalesce(hx.hx_{k},0) AS hx_{k}" for k in CHARLSON)},
       labs.* EXCLUDE (hadm_id)
FROM adm a
JOIN prior pr USING (hadm_id)
LEFT JOIN hx USING (hadm_id)
LEFT JOIN labs USING (hadm_id)
WHERE a.age >= 18
  AND a.dischtime >= a.admittime + INTERVAL 24 HOUR
  AND (a.deathtime IS NULL OR a.deathtime >= a.admittime + INTERVAL 24 HOUR)
) TO '{(DATA / "cohort.parquet").as_posix()}' (FORMAT parquet)
""")

s = con.sql(f"SELECT count(*) n, count(DISTINCT subject_id) pts, avg(label) mort FROM read_parquet('{(DATA / 'cohort.parquet').as_posix()}')").fetchall()[0]
print(f"cohort: {s[0]} admissions, {s[1]} patients, mortality {s[2]:.4f}")
