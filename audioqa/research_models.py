"""Independent local audio embedding and phoneme observations."""
import csv
from pathlib import Path

import numpy as np
from scipy.special import log_softmax, logsumexp

from scripts.common import file_digest, read_json
from scripts.reverse_transcribe import audio_16k


def onnx_session(path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class Yamnet:
    def __init__(self, location):
        model = Path(location) / "yamnet.onnx"
        self.sha256 = file_digest(model)
        self.session = onnx_session(model)
        with (Path(location) / "yamnet_class_map.csv").open(encoding="utf-8", newline="") as handle:
            self.names = {int(row["index"]): row["display_name"] for row in csv.DictReader(handle)}

    def frames(self, values, rate):
        mono = audio_16k(values, rate)
        scores, embeddings, _ = self.session.run(None, {"waveform": mono})
        return scores, embeddings

    def describe(self, values, rate):
        scores, embeddings = self.frames(values, rate)
        pooled = scores.mean(axis=0)
        vector = embeddings.mean(axis=0)
        top = sorted(np.argsort(pooled)[-5:], key=lambda index: -pooled[index])
        return vector, {self.names[index]: float(pooled[index]) for index in top}


def normalized_vectors(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)


def neighbor_distances(vectors, seconds, excluded_s=30, neighbors=5):
    unit = normalized_vectors(vectors)
    if len(unit) < neighbors + 1:
        raise ValueError("Embedding scan requires more frames than nearest neighbors.")
    scores = np.full(len(unit), np.nan, dtype=np.float32)
    for first in range(0, len(unit), 128):
        last = min(first + 128, len(unit))
        similarities = unit[first:last] @ unit.T
        similarities[np.abs(seconds[first:last, None] - seconds) < excluded_s] = -np.inf
        best = np.partition(similarities, -neighbors, axis=1)[:, -neighbors:]
        valid = np.isfinite(best).all(axis=1)
        scores[first:last][valid] = 1 - best[valid].mean(axis=1)
    return scores


def ctc_log_probability(log_probs, tokens, blank):
    """Marginalize every CTC alignment for one phone hypothesis."""
    if not tokens or not len(log_probs):
        return -np.inf
    chain = np.full(len(tokens) * 2 + 1, blank)
    chain[1::2] = tokens
    previous = np.full(len(chain), -np.inf)
    previous[0] = 0
    can_skip = np.r_[False, False, (chain[2:] != blank) & (chain[2:] != chain[:-2])]
    for frame in log_probs:
        move = np.r_[-np.inf, previous[:-1]]
        skip = np.where(can_skip, np.r_[-np.inf, -np.inf, previous[:-2]], -np.inf)
        previous = np.logaddexp(np.logaddexp(previous, move), skip) + frame[chain]
    return float(logsumexp(previous[-2:]))


class Wav2vecPhones:
    step_s = 320 / 16000
    frame_center_s = 199.5 / 16000

    def __init__(self, location):
        location = Path(location)
        model = location / "wav2vec2_int8.onnx"
        self.model_asset = {name: file_digest(location / name) for name in (model.name, "vocab.json")}
        self.session = onnx_session(model)
        self.ids = read_json(location / "vocab.json")
        self.phones = {i: label for label, i in self.ids.items()}
        self.blank = self.ids["<pad>"]

    def posteriors(self, values, rate):
        audio = audio_16k(values, rate)
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        logits, = self.session.run(None, {"input_values": audio[None, :]})
        return log_softmax(logits[0], axis=1)

    def frame_times(self, count, origin):
        return origin + self.frame_center_s + np.arange(count) * self.step_s

    def observe(self, posteriors, origin):
        best = np.argmax(posteriors, axis=1)
        boundaries = np.r_[0, 1 + np.flatnonzero(best[1:] != best[:-1]), len(best)]
        centers = self.frame_times(len(best) + 1, origin) - self.step_s / 2
        rows = []
        for first, last in zip(boundaries, boundaries[1:]):
            phone = self.phones[int(best[first])]
            if best[first] == self.blank or phone.startswith("<"):
                continue
            rows.append({"phone": phone, "start": float(centers[first]), "end": float(centers[last]),
                         "active_frames": int(last - first),
                         "mean_frame_posterior": float(np.exp(posteriors[first:last, best[first]]).mean())})
        return rows

    def compare(self, posteriors, hypotheses):
        scores = {}
        for name, phones in hypotheses.items():
            missing = [phone for phone in phones if phone not in self.ids]
            score = None if missing else ctc_log_probability(posteriors, [self.ids[p] for p in phones], self.blank)
            scores[name] = {"phones": phones, "out_of_vocabulary": missing,
                            "log_ctc_mass": score if score is not None and np.isfinite(score) else None}
        return scores
