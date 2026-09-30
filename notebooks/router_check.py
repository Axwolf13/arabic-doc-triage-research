"""Apply notebook 12's confidence router to all 50 stored Surya outputs.

Notebook 12 validated the router on hand-picked cases. This runs the same rule
(flag a document when its minimum line confidence is below 0.92) over every
sample in surya_50_outputs.json, no OCR engine needed.

    python router_check.py
"""
import json
from pathlib import Path

THRESHOLD = 0.92  # notebook 12

rows = json.loads((Path(__file__).parent / "surya_50_outputs.json").read_text(encoding="utf-8"))
cer = {r["index"]: r["surya_cer"] * 100 for r in rows}  # stored as a fraction
flagged = {r["index"] for r in rows if r["surya_min_conf"] < THRESHOLD}


def report(label, ids):
    caught = sorted(i for i in ids if i in flagged)
    missed = sorted(i for i in ids if i not in flagged)
    print(f"{label}: {len(caught)} of {len(ids)} flagged, missed {missed or 'none'}")


print(f"Flagged {len(flagged)} of {len(rows)} documents at min confidence < {THRESHOLD}")
report("Catastrophic (CER > 30%)", [i for i, c in cer.items() if c > 30])
report("Poor or worse (CER > 10%)", [i for i, c in cer.items() if c > 10])
false_alarms = sorted(i for i in flagged if cer[i] <= 10)
print(f"False alarms (flagged, CER <= 10%): {len(false_alarms)} {false_alarms}")
print(f"Usable (CER <= 10%): {sum(c <= 10 for c in cer.values())} of {len(rows)}, "
      f"of which CER <= 5%: {sum(c <= 5 for c in cer.values())}")
