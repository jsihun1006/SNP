"""Optional metrics for permitted, human-labelled EXTERNAL matches only.

Not used by the provided-match diagnostic. Predictions are never their own truth.
"""
import argparse
import csv
import json
from pathlib import Path

LABELS=("TEAM_A","TEAM_B","REFEREE","UNKNOWN")


def evaluate(records):
    if not records:raise ValueError("No external ground truth supplied; metrics were not calculated")
    matrix={a:{b:0 for b in LABELS} for a in LABELS}
    for r in records:
        game=r.get("game_id","")
        normalized="".join(c for c in game.lower() if c.isalnum())
        if r.get("evaluation_scope")!="external_match" or not game or "barath" in normalized or "20240824" in normalized:
            raise ValueError("Only explicitly external-match labels are permitted; provided-match manual labels are not consumed")
        if r.get("label_source")!="human_external":raise ValueError("Automatic predictions/pseudo-labels cannot serve as evaluation truth")
        truth,pred=r.get("truth"),r.get("prediction")
        if truth not in LABELS or pred not in LABELS:raise ValueError("Invalid evaluation class")
        matrix[truth][pred]+=1
    metrics={}
    for k in LABELS:
        tp=matrix[k][k];fn=sum(matrix[k].values())-tp;fp=sum(matrix[a][k] for a in LABELS)-tp
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
        metrics[k]=dict(precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,support=tp+fn)
    return dict(external_records=len(records),class_metrics=metrics,macro_f1=sum(m["f1"] for m in metrics.values())/len(LABELS),
                confusion_matrix=matrix,known_prediction_coverage=sum(r["prediction"]!="UNKNOWN" for r in records)/len(records),
                provided_match_evaluated=False,zero_division=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--csv",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    with a.csv.open(encoding="utf-8-sig",newline="") as f:result=evaluate(list(csv.DictReader(f)))
    with a.output.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
