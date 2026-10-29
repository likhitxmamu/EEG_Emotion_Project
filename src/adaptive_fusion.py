"""Small validation-reliability and trial-confidence gated fusion layer."""

import math

import numpy as np

from fusion_engine import CLASS_NAMES, FROZEN_CONFIG, FUSION_WEIGHTS, MODEL_NAMES


ADAPTIVE_CONFIG = FROZEN_CONFIG["fusion"]["adaptive"]
EPSILON = float(ADAPTIVE_CONFIG["epsilon"])
FROZEN_RELIABILITY_SCORES = ADAPTIVE_CONFIG["validation_reliability_macro_f1"]
FROZEN_RELIABILITY_PRIOR = ADAPTIVE_CONFIG["reliability_prior_weights"]


def validation_reliability_prior(validation_model_outputs):
    """Use validation macro-F1 as nonnegative model reliability, with smoothing."""
    from fusion_engine import compute_metrics

    scores = {}
    for model_name in MODEL_NAMES:
        records = {
            trial: {"true_class": item["true_class"], "final_emotion": item["predicted_class"]}
            for trial, item in validation_model_outputs[model_name].items()
        }
        scores[model_name] = compute_metrics(records)["macro_f1"]
    for name in MODEL_NAMES:
        if not np.isclose(scores[name], FROZEN_RELIABILITY_SCORES[name], atol=1e-12, rtol=0.0):
            raise RuntimeError(
                f"Validation reliability for {name} differs from the frozen configuration: "
                f"{scores[name]} != {FROZEN_RELIABILITY_SCORES[name]}"
            )
    raw = {name: max(0.0, scores[name]) + EPSILON for name in MODEL_NAMES}
    total = sum(raw.values())
    derived_prior = {name: raw[name] / total for name in MODEL_NAMES}
    for name in MODEL_NAMES:
        if not np.isclose(derived_prior[name], FROZEN_RELIABILITY_PRIOR[name], atol=1e-12, rtol=0.0):
            raise RuntimeError(
                f"Derived reliability weight for {name} differs from the frozen configuration"
            )
    return dict(FROZEN_RELIABILITY_PRIOR), scores


def trial_gate(probabilities):
    """Higher normalized certainty modestly increases a model's trial weight."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(CLASS_NAMES),) or not np.isfinite(p).all():
        raise ValueError("Adaptive fusion expects a finite four-class probability vector")
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-5):
        raise ValueError("Adaptive fusion input probabilities must be nonnegative and sum to one")
    probability_floor = float(ADAPTIVE_CONFIG["entropy_probability_floor"])
    entropy = -float(np.sum(np.where(p > 0, p * np.log(np.maximum(p, probability_floor)), 0.0)))
    lower, upper = ADAPTIVE_CONFIG["certainty_clip_range"]
    certainty = max(float(lower), min(float(upper), 1.0 - entropy / math.log(len(CLASS_NAMES))))
    return EPSILON + certainty, entropy, certainty


def fuse_adaptively(split, model_outputs, reliability_prior):
    """Fuse trial-averaged model probabilities with fixed prior and input gate."""
    expected = set(next(iter(model_outputs.values())))
    if set(model_outputs) != set(MODEL_NAMES):
        raise RuntimeError("Adaptive fusion requires exactly the four agreed branches")
    if set(reliability_prior) != set(MODEL_NAMES):
        raise RuntimeError("Reliability prior keys do not match the four branches")
    if any(value < 0 or not np.isfinite(value) for value in reliability_prior.values()):
        raise RuntimeError("Reliability prior must be finite and nonnegative")
    if not np.isclose(sum(reliability_prior.values()), 1.0, atol=1e-10):
        raise RuntimeError("Reliability prior must sum to one")
    for name in MODEL_NAMES:
        if not np.isclose(reliability_prior[name], FROZEN_RELIABILITY_PRIOR[name], atol=1e-12, rtol=0.0):
            raise RuntimeError("Adaptive fusion received weights different from the frozen configuration")

    fused_trials = {}
    for trial_id in expected:
        model_items = {}
        raw_weights = {}
        entropies = {}
        certainties = {}
        for model_name in MODEL_NAMES:
            item = model_outputs[model_name][trial_id]
            probability = np.asarray(item["probabilities"], dtype=np.float64)
            gate, entropy, certainty = trial_gate(probability)
            raw_weights[model_name] = reliability_prior[model_name] * gate
            entropies[model_name] = entropy
            certainties[model_name] = certainty
            model_items[model_name] = {
                "window_count": item["window_count"],
                "probabilities": probability.astype(float).tolist(),
                "predicted_class": item["predicted_class"],
                "model_confidence_score_not_calibrated": item["model_confidence_score_not_calibrated"],
                "window_outputs": item["window_outputs"],
            }
        weight_sum = sum(raw_weights.values())
        weights = {name: value / weight_sum for name, value in raw_weights.items()}
        vector = sum(
            weights[name] * np.asarray(model_outputs[name][trial_id]["probabilities"], dtype=np.float64)
            for name in MODEL_NAMES
        )
        vector /= vector.sum()
        prediction = int(np.argmax(vector))
        first_model = MODEL_NAMES[0]
        fused_trials[trial_id] = {
            "true_class": model_outputs[first_model][trial_id]["true_class"],
            "individual_models": model_items,
            "reliability_prior_weights": reliability_prior,
            "trial_adaptive_weights": weights,
            "normalized_entropy_by_model": entropies,
            "certainty_gate_by_model": certainties,
            "fused_probabilities": vector.astype(float).tolist(),
            "final_emotion": CLASS_NAMES[prediction],
            "model_confidence_score_not_calibrated": float(vector[prediction]),
            "window_count_by_model": {name: model_outputs[name][trial_id]["window_count"] for name in MODEL_NAMES},
        }
    return fused_trials


def equal_weight_fuse(split, model_outputs):
    """Local equal baseline so adaptive and equal use the same fresh predictions."""
    fused_trials = {}
    trial_ids = set(next(iter(model_outputs.values())))
    for trial_id in trial_ids:
        vector = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        model_items = {}
        for model_name in MODEL_NAMES:
            item = model_outputs[model_name][trial_id]
            probabilities = np.asarray(item["probabilities"], dtype=np.float64)
            vector += FUSION_WEIGHTS[model_name] * probabilities
            model_items[model_name] = {
                "window_count": item["window_count"],
                "probabilities": probabilities.tolist(),
                "predicted_class": item["predicted_class"],
                "model_confidence_score_not_calibrated": item["model_confidence_score_not_calibrated"],
                "window_outputs": item["window_outputs"],
            }
        vector /= vector.sum()
        prediction = int(np.argmax(vector))
        fused_trials[trial_id] = {
            "true_class": model_outputs[MODEL_NAMES[0]][trial_id]["true_class"],
            "individual_models": model_items,
            "fusion_weights": FUSION_WEIGHTS,
            "fused_probabilities": vector.tolist(),
            "final_emotion": CLASS_NAMES[prediction],
            "model_confidence_score_not_calibrated": float(vector[prediction]),
        }
    return fused_trials
