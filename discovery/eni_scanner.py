"""
ENI Scanner

Utility to scan CSVs for ENI-like vessel identifiers using known country prefixes.
Detects:
- 8-digit numeric values whose first 3 digits match official ENI prefixes
- 7-digit values starting with 60-69, treated as Belgian ENIs by padding a leading 0

Outputs columns and files with hit counts and top prefixes observed.
"""

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 3-digit ENI prefixes
VALID_PREFIXES = {
    '001','019','020','039','040','059','060','069','070','079','080','099','100','119',
    '120','139','140','159','160','169','170','179','180','189','190','199','200','219',
    '220','239','240','259','260','269','270','279','280','289','290','299','300','309',
    '310','319','320','329','330','339','340','349','350','359','360','369','370','379',
    '380','399','400','419','420','439','440','449','450','459','460','469','470','479',
    '480','489','490','499','500','519','520','539','540','549','550','559','560','569',
    '570','579','580','589','590','599','600','619','620','639','640','649','650','659',
    '660','669','670','679','680','699','700','719','720','739','740','759','760','769',
    '770','799','800','809','810','819','820','829','830','839','840','849','850','859',
    '860','869','870','999'
}

ENI_8_RE = re.compile(r'^\d{8}$')
ENI_7_BE_RE = re.compile(r'^(6[0-9])\d{5}$')  # 7-digit starting 60-69 (Belgium)
ENI_7_ANY_RE = re.compile(r'^\d{7}$')        # generic 7-digit numeric (missing leading 0)


def prefix_to_country(prefix: str) -> str:
    """Map ENI prefix to country name (best effort for known ranges)."""
    mapping = {
        '001': 'France', '019': 'France',
        '020': 'Netherlands', '039': 'Netherlands',
        '040': 'Germany', '059': 'Germany',
        '060': 'Belgium', '069': 'Belgium',
        '070': 'Switzerland', '079': 'Switzerland',
        '080': 'Reserved',
        '100': 'Norway', '119': 'Norway',
        '120': 'Denmark', '139': 'Denmark',
        '140': 'United Kingdom', '159': 'United Kingdom',
        '160': 'Iceland', '169': 'Iceland',
        '170': 'Ireland', '179': 'Ireland',
        '180': 'Portugal', '189': 'Portugal',
        '200': 'Luxembourg', '219': 'Luxembourg',
        '220': 'Finland', '239': 'Finland',
        '240': 'Poland', '259': 'Poland',
        '260': 'Estonia', '269': 'Estonia',
        '270': 'Lithuania', '279': 'Lithuania',
        '280': 'Latvia', '289': 'Latvia',
        '300': 'Austria', '309': 'Austria',
        '310': 'Liechtenstein', '319': 'Liechtenstein',
        '320': 'Czech Republic', '329': 'Czech Republic',
        '330': 'Slovakia', '339': 'Slovakia',
        '350': 'Croatia', '359': 'Croatia',
        '360': 'Serbia', '369': 'Serbia',
        '370': 'Bosnia and Herzegovina', '379': 'Bosnia and Herzegovina',
        '380': 'Hungary', '399': 'Hungary',
        '400': 'Russia', '419': 'Russia',
        '420': 'Ukraine', '439': 'Ukraine',
        '440': 'Belarus', '449': 'Belarus',
        '450': 'Moldova', '459': 'Moldova',
        '460': 'Romania', '469': 'Romania',
        '470': 'Bulgaria', '479': 'Bulgaria',
        '480': 'Georgia', '489': 'Georgia',
        '500': 'Turkey', '519': 'Turkey',
        '520': 'Greece', '539': 'Greece',
        '540': 'Cyprus', '549': 'Cyprus',
        '550': 'Albania', '559': 'Albania',
        '560': 'North Macedonia', '569': 'North Macedonia',
        '570': 'Slovenia', '579': 'Slovenia',
        '580': 'Montenegro', '589': 'Montenegro',
        '600': 'Italy', '619': 'Italy',
        '620': 'Spain', '639': 'Spain',
        '640': 'Andorra', '649': 'Andorra',
        '650': 'Malta', '659': 'Malta',
        '660': 'Monaco', '669': 'Monaco',
        '670': 'San Marino', '679': 'San Marino',
        '700': 'Sweden', '719': 'Sweden',
        '720': 'Canada', '739': 'Canada',
        '740': 'United States', '759': 'United States',
        '760': 'Israel', '769': 'Israel',
        '800': 'Azerbaijan', '809': 'Azerbaijan',
        '810': 'Kazakhstan', '819': 'Kazakhstan',
        '820': 'Kyrgyzstan', '829': 'Kyrgyzstan',
        '830': 'Tajikistan', '839': 'Tajikistan',
        '840': 'Turkmenistan', '849': 'Turkmenistan',
        '850': 'Uzbekistan', '859': 'Uzbekistan',
        '860': 'Iran', '869': 'Iran',
    }
    return mapping.get(prefix, 'Reserved/Other')


def classify_eni(val: str) -> Tuple[Optional[str], bool]:
    """
    Return ENI prefix if value matches ENI patterns, else None.
    - 8-digit: return first 3 digits if in VALID_PREFIXES
    - 7-digit starting 60-69: pad leading 0 (e.g., 60 -> 060) and return if valid
    - 7-digit other: treat as missing leading 0 and use first 2 digits with leading 0
    """
    v = val.strip()
    if ENI_8_RE.match(v):
        pref = v[:3]
        return (pref if pref in VALID_PREFIXES else None, False)
    m = ENI_7_BE_RE.match(v)
    if m:
        pref = '0' + m.group(1)  # e.g., 60 -> 060
        return (pref if pref in VALID_PREFIXES else None, True)
    if ENI_7_ANY_RE.match(v):
        pref = '0' + v[:2]  # assume missing leading 0 before country code
        if pref in VALID_PREFIXES:
            return (pref, True)
    return (None, False)


def scan_data_dir(
    data_dir: Path = Path("data"),
    max_rows: int = 2000000,
    delimiter: str = ';',
    encoding: str = 'utf-8-sig',
    min_ratio: float = 0.0,
    max_values: int = 5000000
) -> Tuple[
    List[Tuple[str, str, int, int, float, float, int, Tuple[str, int, str], Tuple[str, int, str], Tuple[str, int, str], bool, int, float]],
    Dict[Tuple[str, str], set]
]:
    """
    Scan CSVs in data_dir for ENI-like values.

    Returns:
        results: list of tuples (file, column, hits, hits7, hit_ratio, hit_ratio_nonblank, rows_seen,
                                 top1, top2, top3, seven_flag, be_hits, be_ratio_nonblank)
        value_sets: mapping (file, column) -> set of non-blank values (capped)
    """
    results = []
    value_sets: Dict[Tuple[str, str], set] = {}

    for csv_path in sorted(data_dir.glob("*.csv")):
        try:
            with csv_path.open(encoding=encoding, newline="") as f:
                reader = csv.reader(f, delimiter=delimiter)
                headers = next(reader, [])
                counters = [Counter() for _ in headers]
                rows_seen = 0
                non_blank = [0] * len(headers)
                seven_hits = [0] * len(headers)
                col_sets = [set() for _ in headers]
                for row in reader:
                    rows_seen += 1
                    for i, val in enumerate(row):
                        pref, is_seven = classify_eni(val)
                        if pref:
                            counters[i][pref] += 1
                            if is_seven:
                                seven_hits[i] += 1
                        if val.strip():
                            non_blank[i] += 1
                            if len(col_sets[i]) < max_values:
                                col_sets[i].add(val.strip())
                    if rows_seen >= max_rows:
                        break
        except Exception as e:
            print(f"Skip {csv_path.name}: {e}")
            continue

        for idx, c in enumerate(counters):
            if not c:
                continue
            total_hits = sum(c.values())
            top_items = c.most_common(3)
            while len(top_items) < 3:
                top_items.append(('', 0))
            top_triplets = []
            for p, n in top_items[:3]:
                country = prefix_to_country(p) if p else ''
                top_triplets.append((p, n, country))
            be_hits = sum(n for p, n in c.items() if prefix_to_country(p) == 'Belgium')
            col_name = headers[idx] if idx < len(headers) else f"col{idx}"
            hit_ratio = total_hits / rows_seen if rows_seen else 0
            non_blank_total = non_blank[idx] if idx < len(non_blank) else rows_seen
            hit_ratio_nonblank = total_hits / non_blank_total if non_blank_total else 0
            be_ratio_nonblank = be_hits / non_blank_total if non_blank_total else 0
            if hit_ratio >= min_ratio or hit_ratio_nonblank >= min_ratio:
                seven_flag = seven_hits[idx] > (total_hits * 0.5) if total_hits > 0 else False
                results.append((csv_path.name, col_name, total_hits, seven_hits[idx], hit_ratio, hit_ratio_nonblank, rows_seen,
                                top_triplets[0], top_triplets[1], top_triplets[2], seven_flag, be_hits, be_ratio_nonblank))
            # always cache values for overlap use
            value_sets[(csv_path.name, col_name)] = col_sets[idx]

    results_sorted = sorted(results, key=lambda x: (-x[2], x[0], x[1]))
    return results_sorted, value_sets


def main():
    """CLI entry point."""
    import argparse
    import csv as csvlib

    parser = argparse.ArgumentParser(description="Scan CSVs for ENI-like vessel identifiers.")
    parser.add_argument("--data-dir", default="data", help="Directory containing CSV files (default: data)")
    parser.add_argument("--max-rows", type=int, default=200000, help="Max rows to sample per file (default: 200000)")
    parser.add_argument("--delimiter", default=';', help="CSV delimiter (default: ;)") 
    parser.add_argument("--encoding", default='utf-8-sig', help="CSV encoding (default: utf-8-sig)")
    parser.add_argument("--min-ratio", type=float, default=0.0, help="Minimum hit ratio (overall or non-blank) to include a column (default: 0.0)")
    parser.add_argument("--out", default="discovery/cache/eni_scanner_output.csv", help="Output CSV path (default: discovery/cache/eni_scanner_output.csv)")
    parser.add_argument("--be-threshold", type=float, default=0.02, help="Minimum BE hit ratio (non-blank) to trigger overlap analysis (default: 0.01)")
    parser.add_argument("--overlap-threshold", type=float, default=0.0, help="Minimum overlap ratio to report candidate matches (default: 0.0)")
    parser.add_argument("--unique-max-ratio", type=float, default=0.0, help="Max unique/rows ratio to list values (default: 0.0)")
    parser.add_argument("--unique-max-values", type=int, default=200, help="Max unique values to list (default: 200)")
    args = parser.parse_args()

    results, value_sets = scan_data_dir(Path(args.data_dir), max_rows=args.max_rows, delimiter=args.delimiter, encoding=args.encoding, min_ratio=args.min_ratio)
    if not results:
        print("No ENI-like values found.")
        return

    header = (
        f"{'file':30} {'column':25} {'rows':<8} {'hits':<8} {'7digit':<8} "
        f"{'hit_ratio':<10} {'hit_ratio_nonblank':<18} "
        f"{'top1_prefix':<12} {'top1_hits':<10} {'top1_country':<15} "
        f"{'top2_prefix':<12} {'top2_hits':<10} {'top2_country':<15} "
        f"{'top3_prefix':<12} {'top3_hits':<10} {'top3_country':<15} {'missing0?':<10}"
    )
    print(header)

    # Prepare rows for CSV
    rows_out = []
    for file, col, hits, hits7, ratio, ratio_nb, rows, top1, top2, top3, seven_flag, be_hits, be_ratio_be in results:
        def fmt_triplet(t):
            p, n, c = t
            if c == 'Belgium':
                return f"{p} (BE)", f"{n} (BE)", f"{c} (BE)"
            return p, str(n), c
        p1, n1, c1 = fmt_triplet(top1)
        p2, n2, c2 = fmt_triplet(top2)
        p3, n3, c3 = fmt_triplet(top3)
        print(
            f"{file:30} {col:25} {rows:<8} {hits:<8} {hits7:<8} {ratio:.1%}     {ratio_nb:.1%}           "
            f"{p1:<12} {n1:<10} {c1:<15} "
            f"{p2:<12} {n2:<10} {c2:<15} "
            f"{p3:<12} {n3:<10} {c3:<15} {('YES' if seven_flag else 'NO'):<10}"
        )
        rows_out.append([
            file, col, rows, hits, hits7, f"{ratio:.4f}", f"{ratio_nb:.4f}",
            p1, n1, c1, p2, n2, c2, p3, n3, c3, 'YES' if seven_flag else 'NO',
            be_hits, f"{be_ratio_be:.4f}"
        ])

    # Write CSV (overwrite)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csvlib.writer(f)
        writer.writerow([
            "file", "column", "rows", "hits", "hits_7digit", "hit_ratio", "hit_ratio_nonblank",
            "top1_prefix", "top1_hits", "top1_country",
            "top2_prefix", "top2_hits", "top2_country",
            "top3_prefix", "top3_hits", "top3_country",
            "missing_leading_zero_hint",
            "be_hits", "be_ratio_nonblank"
        ])
        writer.writerows(rows_out)
    print(f"\nWrote results to {out_path}")

    # Overlap analysis for BE-significant columns
    candidates = [
        (file, col, hits, hits7, ratio, ratio_nb, rows, top1, top2, top3, seven_flag, be_hits, be_ratio_be)
        for (file, col, hits, hits7, ratio, ratio_nb, rows, top1, top2, top3, seven_flag, be_hits, be_ratio_be) in results
        if be_ratio_be >= args.be_threshold and be_hits > 0
    ]

    overlap_rows = []
    for cand in candidates:
        c_file, c_col = cand[0], cand[1]
        key_c = (c_file, c_col)
        vals_c = value_sets.get(key_c, set())
        len_c = len(vals_c)
        if len_c == 0:
            continue
        for (f2, c2), vals2 in value_sets.items():
            if (f2 == c_file and c2 == c_col) or len(vals2) == 0:
                continue
            intersect = vals_c & vals2
            if not intersect:
                continue
            ratio_overlap = len(intersect) / min(len_c, len(vals2))
            if ratio_overlap >= args.overlap_threshold:
                overlap_rows.append([
                    c_file, c_col, len_c,
                    f2, c2, len(vals2),
                    len(intersect), f"{ratio_overlap:.4f}"
                ])

    if overlap_rows:
        overlap_path = Path("discovery/cache/eni_scanner_links.csv")
        overlap_path.parent.mkdir(parents=True, exist_ok=True)
        with overlap_path.open("w", newline="", encoding="utf-8") as f:
            writer = csvlib.writer(f)
            writer.writerow([
                "candidate_file", "candidate_column", "candidate_unique_values",
                "match_file", "match_column", "match_unique_values",
                "overlap_count", "overlap_ratio"
            ])
            writer.writerows(overlap_rows)
        print(f"Wrote overlap candidates to {overlap_path}")

    # Unique value listing for BE-significant columns (only those columns)
    file_rows = {}
    for file, _, _, _, _, _, rows, *_ in results:
        file_rows[file] = max(file_rows.get(file, 0), rows)

    unique_rows = []
    for cand in candidates:
        c_file, c_col = cand[0], cand[1]
        rows_total = file_rows.get(c_file, args.max_rows)
        # For every column in this file, compute uniques (subject to size bounds)
        for (f2, c2), vals in value_sets.items():
            if f2 != c_file:
                continue
            uniq_count = len(vals)
            if rows_total == 0:
                continue
            ratio = uniq_count / rows_total
            sample_vals = ""
            if ratio <= args.unique_max_ratio and uniq_count <= args.unique_max_values:
                sample_vals = "|".join(sorted(list(vals))[:args.unique_max_values])
            unique_rows.append([
                c_file, c_col, f2, c2, rows_total, uniq_count, f"{ratio:.4f}", sample_vals
            ])

    if unique_rows:
        uniques_path = Path("discovery/cache/eni_scanner_uniques.csv")
        uniques_path.parent.mkdir(parents=True, exist_ok=True)
        with uniques_path.open("w", newline="", encoding="utf-8") as f:
            writer = csvlib.writer(f)
            writer.writerow([
                "be_candidate_file", "be_candidate_column",
                "file", "column", "rows", "unique_count", "unique_ratio", "values"
            ])
            writer.writerows(unique_rows)
        print(f"Wrote unique value candidates to {uniques_path}")


if __name__ == "__main__":
    main()
