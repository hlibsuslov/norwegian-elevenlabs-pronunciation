"""Find short near-zero runs bounded by former and following signal."""
import numpy as np

from .segment_audio import db_level, positive_runs


def dropouts(samples, rate):
    result = []
    margin = round(0.025 * rate)
    quiet = np.max(np.abs(samples), axis=1) < 1e-5
    for start, end in positive_runs(quiet):
        if start < margin or end+margin > len(samples) or not 0.008 <= (end-start)/rate <= 0.35:
            continue
        sides = [float(max(db_level(samples[a:b]))) for a,b in [(start-margin,start), (end,end+margin)]]
        edge = max(np.max(np.abs(samples[start]-samples[start-1])), np.max(np.abs(samples[end]-samples[end-1])))
        if min(sides) > -32 and edge > 0.04:
            result.append(("dropout", start, end, {"neighbor_dbfs": sides, "edge_step": float(edge)}))
    return result
