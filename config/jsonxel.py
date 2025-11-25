import json
import pandas as pd

INPUT_FILE = "relationship_metadata.json"
OUTPUT_XLSX = "relationship_metadata.xlsx"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---------- 1) relationships → wide table ----------
rel_rows = []

for table_name, cols in data.get("relationships", {}).items():
    for col_name, props in cols.items():
        row = {
            "table": table_name,
            "column": col_name,
        }
        # props contains: target_table, target_column, discovery_method, confidence, ...
        row.update(props)
        rel_rows.append(row)

df_relationships = pd.DataFrame(rel_rows).sort_values(
    ["table", "column", "target_table", "target_column"]
)

# ---------- 2) junction_tables → bridge metadata ----------
junction_rows = []

for jt_name, props in data.get("junction_tables", {}).items():
    # Extract temporal if present
    temporal = props.get("temporal", {})
    base = {k: v for k, v in props.items() if k not in ["temporal"]}

    row = {
        "junction_table": jt_name,
        "type": base.get("type"),
        "links": ",".join(base.get("links", [])) if base.get("links") else None,
        "description": base.get("description"),
        "temporal_valid_from": temporal.get("valid_from"),
        "temporal_valid_until": temporal.get("valid_until"),
        "temporal_null_means_active": temporal.get("null_means_active"),
        "temporal_description": temporal.get("description"),
    }
    junction_rows.append(row)

df_junctions = pd.DataFrame(junction_rows).sort_values("junction_table")

# ---------- 3) data_quality → one table per level ----------
dq_rows = []
dq_null_rows = []  # exploded high_null_columns

for table_name, props in data.get("data_quality", {}).items():
    dq_rows.append({
        "table": table_name,
        "row_count": props.get("row_count"),
        "completeness_score": props.get("completeness_score"),
        "is_sample": props.get("is_sample"),
    })

    high_null = props.get("high_null_columns", {})
    for col_name, col_stats in high_null.items():
        dq_null_rows.append({
            "table": table_name,
            "column": col_name,
            "null_rate": col_stats.get("null_rate"),
            "impact": col_stats.get("impact"),
        })

df_dq_tables = pd.DataFrame(dq_rows).sort_values("table")
df_dq_nulls = pd.DataFrame(dq_null_rows).sort_values(["table", "column"]) if dq_null_rows else pd.DataFrame()

# ---------- 4) metadata → metadata sheet ----------
meta = data.get("metadata", {})
df_metadata = pd.DataFrame(
    [{"key": k, "value": v} for k, v in meta.items()]
)

# ---------- 5) Write to Excel ----------
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    df_metadata.to_excel(writer, sheet_name="metadata", index=False)
    df_relationships.to_excel(writer, sheet_name="relationships", index=False)
    df_junctions.to_excel(writer, sheet_name="junction_tables", index=False)
    df_dq_tables.to_excel(writer, sheet_name="data_quality_tables", index=False)
    if not df_dq_nulls.empty:
        df_dq_nulls.to_excel(writer, sheet_name="data_quality_nulls", index=False)

print(f"Exported to {OUTPUT_XLSX}")
