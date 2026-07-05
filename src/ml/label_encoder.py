"""
Centralized direction label encoding for ML pipelines.

Trading semantics use [-1, 0, 1] (sell, hold, buy).
XGBoost/sklearn classifiers require contiguous classes [0, 1, 2].

This module provides the single source of truth for all label
encoding/decoding across training, inference, evaluation, replay,
and online learning paths.

Usage:
    from src.ml.label_encoder import DirectionEncoder

    # Training: convert trading labels to model classes
    y_encoded = DirectionEncoder.encode(y_raw)

    # Inference: convert model predictions back to trading labels
    direction = DirectionEncoder.decode(pred_class)

    # Probability indexing
    prob_down = proba[DirectionEncoder.DOWN_CLASS]
    prob_flat = proba[DirectionEncoder.FLAT_CLASS]
    prob_up   = proba[DirectionEncoder.UP_CLASS]
"""

from __future__ import annotations

import numpy as np


class DirectionEncoder:
    """Bidirectional mapping between trading direction labels and model classes.

    Trading Labels (domain semantics):
        -1 = Down / Sell
         0 = Flat / Hold
         1 = Up / Buy

    Model Classes (XGBoost/sklearn):
         0 = Down / Sell
         1 = Flat / Hold
         2 = Up / Buy
    """

    # ── Trading label constants ──────────────────────────────────────────
    SELL_LABEL: int = -1
    HOLD_LABEL: int = 0
    BUY_LABEL: int = 1

    # ── Model class constants ────────────────────────────────────────────
    DOWN_CLASS: int = 0
    FLAT_CLASS: int = 1
    UP_CLASS: int = 2

    NUM_CLASSES: int = 3

    # ── Mapping dictionaries ────────────────────────────────────────────
    LABEL_TO_CLASS: dict[int, int] = {
        -1: 0,  # sell → class 0
         0: 1,  # hold → class 1
         1: 2,  # buy  → class 2
    }

    CLASS_TO_LABEL: dict[int, int] = {
        0: -1,  # class 0 → sell
        1:  0,  # class 1 → hold
        2:  1,  # class 2 → buy
    }

    # Human-readable names for each class
    CLASS_NAMES: dict[int, str] = {
        0: "DOWN",
        1: "FLAT",
        2: "UP",
    }

    LABEL_NAMES: dict[int, str] = {
        -1: "SELL",
         0: "HOLD",
         1: "BUY",
    }

    @classmethod
    def encode(cls, labels: np.ndarray) -> np.ndarray:
        """Convert trading direction labels [-1, 0, 1] to model classes [0, 1, 2].

        Args:
            labels: Array of trading direction labels.

        Returns:
            Array of model class indices suitable for XGBoost/sklearn.

        Raises:
            ValueError: If labels contain values outside {-1, 0, 1}.
        """
        labels = np.asarray(labels)
        unique = set(np.unique(labels).tolist())
        valid = {-1, 0, 1}
        if not unique.issubset(valid):
            invalid = unique - valid
            raise ValueError(
                f"Invalid direction labels: {invalid}. Expected subset of {valid}."
            )
        return (labels + 1).astype(int)

    @classmethod
    def decode(cls, classes: np.ndarray | int) -> np.ndarray | int:
        """Convert model class predictions [0, 1, 2] to trading labels [-1, 0, 1].

        Args:
            classes: Array or scalar of model class predictions.

        Returns:
            Trading direction labels (-1=sell, 0=hold, 1=buy).
        """
        if isinstance(classes, (int, np.integer)):
            return cls.CLASS_TO_LABEL[int(classes)]
        classes = np.asarray(classes)
        return (classes - 1).astype(int)

    @classmethod
    def encode_scalar(cls, label: int) -> int:
        """Encode a single trading label to model class."""
        return cls.LABEL_TO_CLASS[label]

    @classmethod
    def decode_scalar(cls, model_class: int) -> int:
        """Decode a single model class to trading label."""
        return cls.CLASS_TO_LABEL[model_class]

    @classmethod
    def class_name(cls, model_class: int) -> str:
        """Get human-readable name for a model class index."""
        return cls.CLASS_NAMES.get(model_class, f"UNKNOWN({model_class})")

    @classmethod
    def validate_classes(cls, y: np.ndarray) -> bool:
        """Check that an array contains only valid model classes [0, 1, 2]."""
        unique = set(np.unique(y).tolist())
        return unique.issubset({0, 1, 2})

    @classmethod
    def validate_labels(cls, y: np.ndarray) -> bool:
        """Check that an array contains only valid trading labels [-1, 0, 1]."""
        unique = set(np.unique(y).tolist())
        return unique.issubset({-1, 0, 1})
