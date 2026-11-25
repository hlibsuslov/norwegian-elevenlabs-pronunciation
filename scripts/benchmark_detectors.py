"""Match predictions to manually annotated categories and calculate held-out metrics."""
from collections import Counter
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment


def separation(left,right):
    return max(0,left["start"]-right["end"],right["start"]-left["end"])


def union_duration(intervals):
    length=0.0
    end=0.0
    for start,stop in sorted(intervals):
        length+=max(0,stop-max(start,end))
        end=max(end,stop)
    return length


def benchmark(events, truth, tolerance=.30):
    if tolerance<0 or any(not 0<=row["start"]<row["end"] or not row.get("category") for row in truth):
        raise ValueError("Annotations need an exact category and valid interval.")
    candidates=[row for row in events if row.get("severity")!="INFO"]
    losses=np.full((len(truth),len(candidates)),1e6)
    for i, target in enumerate(truth):
        for j,event in enumerate(candidates):
            if event["category"]==target["category"] and separation(event,target)<=tolerance:
                losses[i,j]=abs(event["start"]-target["start"])
    truth_ids,event_ids=linear_sum_assignment(losses)
    pairs=[(int(i),int(j),float(losses[i,j])) for i,j in zip(truth_ids,event_ids) if losses[i,j]<1e6]
    tp=len(pairs)
    fp,fn=len(candidates)-tp,len(truth)-tp
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    per_category=Counter(truth[i]["category"] for i,j,d in pairs)
    truth_counts=Counter(row["category"] for row in truth)
    return {
        "tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,
        "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        "timestamp_mean_absolute_start_error_s":float(np.mean([d for i,j,d in pairs])) if pairs else None,
        "category_recall":{name:per_category[name]/count for name,count in truth_counts.items()},
        "manual_review_minutes":union_duration((e.get("clip_start",e["start"]),e.get("clip_end",e["end"]))
                                               for e in candidates)/60,
        "matches":[{"annotation_index":i,"event_id":candidates[j]["id"],"start_error_s":d} for i,j,d in pairs],
    }
