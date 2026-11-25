import numpy as np
import soundfile as sf
import pytest

from audioqa.research_benchmark import evaluate
from audioqa.research_models import ctc_log_probability, neighbor_distances
from audioqa.research_review import ReviewDataset
from scripts.common import write_json


def test_approximate_matching_uses_cardinality_and_separates_unreviewed_candidates():
    targets = [{"id": "1", "point_s": 10, "category": "artifact"},
               {"id": "2", "point_s": 12, "category": "artifact"},
               {"id": "3", "point_s": 15, "category": "pronunciation"}]
    candidates = [{"id": "a", "start": 8.9, "end": 9.1, "category": "click_or_pop"},
                  {"id": "b", "start": 10.1, "end": 10.3, "category": "transient_outlier"},
                  {"id": "c", "start": 14.9, "end": 15.1, "category": "embedding_outlier"}]
    result = evaluate(targets, candidates, 30, 2)
    assert [(row["annotation_id"], row["event_id"]) for row in result["typed_matches"]] == [("1", "a"), ("2", "b")]
    assert result["category_disagreements"][0]["event_id"] == "c"
    assert result["unadjudicated_candidates"] == ["c"]
    assert result["precision"] is None and result["f1"] is None


def test_reviewed_metrics_use_actual_intervals_and_dismissals():
    label = {"id": "p", "start": 1., "end": 1.2, "category": "artifact", "status": "CONFIRMED"}
    hit = {"id": "h", "start": 1.1, "end": 1.25, "category": "clipping"}
    miss = {"id": "m", "start": 6., "end": 6.1, "category": "click_or_pop"}
    pending = evaluate([label], [hit, miss], 1800, .2, points=False, complete=True)
    assert pending["precision"] is None
    result = evaluate([label], [hit, miss], 1800, .2, points=False, complete=True,
                      candidate_decisions={"m": "HARD_NEGATIVE"})
    assert result["precision"] == .5
    assert result["false_positives_per_hour"] == 2
    assert result["start_mae_s"] == pytest.approx(.1)


def test_ctc_sum_handles_a_repeat_and_zero_eligible_neighbors():
    frame_probs = np.log([[.2, .7, .1], [.8, .1, .1], [.1, .8, .1]])
    assert np.exp(ctc_log_probability(frame_probs, [1, 1], 0)) == pytest.approx(.448)
    vectors = np.eye(6)
    seconds = np.arange(6)
    assert np.isnan(neighbor_distances(vectors, seconds)).all()


def test_review_decision_preserves_history_and_selected_audio(tmp_path):
    rate = 8000
    source = np.linspace(-.4, .4, rate, dtype=np.float32)
    proxy = tmp_path / "decoded.wav"
    sf.write(proxy, source, rate, subtype="FLOAT")
    sha = "a"*64
    event = {"id": "demo", "start": .2, "end": .3, "source_sha256": sha, "audio_file": "decoded.wav"}
    report = {"dataset_review": True, "review_run": "recording", "events": [event], "source": {"sha256": sha},
              "sample_rate": rate, "duration_s": 1.}
    write_json(tmp_path/"metadata"/"recording-manifest.json", {"source": {"sha256": sha}})
    dataset = ReviewDataset(tmp_path, report, proxy)
    decision = {"id": "demo", "status": "CONFIRMED", "category": "clipping", "recording_id": "recording",
                "start": .21, "end": .3, "reviewed_at": "2026-01-01T10:00:00Z", "comments": "Checked"}
    stored = dataset.save(decision)
    selected, actual_rate = sf.read(tmp_path/stored["history"][0]["audio"], dtype="float32")
    assert actual_rate == rate and np.array_equal(selected, source[1680:2400])
    assert len(dataset.save(decision)["history"]) == 1
    revised = dataset.save({**decision, "reviewed_at": "2026-01-01T11:00:00Z", "status": "HARD_NEGATIVE",
                            "category": "natural_fricative"})
    assert revised["partition"] == "hard_negatives"
    assert len(revised["history"]) == 2
    with pytest.raises(ValueError, match="recording ID"):
        dataset.save({**decision, "recording_id": "other"})
