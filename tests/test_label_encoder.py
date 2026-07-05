"""Tests for DirectionEncoder — centralized label encoding."""

import numpy as np
import pytest

from src.ml.label_encoder import DirectionEncoder


class TestDirectionEncoder:
    """Verify bidirectional label mapping."""

    def test_encode_basic(self):
        """Trading labels [-1, 0, 1] → model classes [0, 1, 2]."""
        labels = np.array([-1, 0, 1, -1, 1, 0])
        encoded = DirectionEncoder.encode(labels)
        expected = np.array([0, 1, 2, 0, 2, 1])
        np.testing.assert_array_equal(encoded, expected)

    def test_decode_basic(self):
        """Model classes [0, 1, 2] → trading labels [-1, 0, 1]."""
        classes = np.array([0, 1, 2, 0, 2, 1])
        decoded = DirectionEncoder.decode(classes)
        expected = np.array([-1, 0, 1, -1, 1, 0])
        np.testing.assert_array_equal(decoded, expected)

    def test_roundtrip(self):
        """encode(decode(x)) == x and decode(encode(x)) == x."""
        labels = np.array([-1, 0, 1, 0, -1, 1])
        assert np.array_equal(DirectionEncoder.decode(DirectionEncoder.encode(labels)), labels)

        classes = np.array([0, 1, 2, 1, 0, 2])
        assert np.array_equal(DirectionEncoder.encode(DirectionEncoder.decode(classes)), classes)

    def test_encode_scalar(self):
        assert DirectionEncoder.encode_scalar(-1) == 0
        assert DirectionEncoder.encode_scalar(0) == 1
        assert DirectionEncoder.encode_scalar(1) == 2

    def test_decode_scalar(self):
        assert DirectionEncoder.decode_scalar(0) == -1
        assert DirectionEncoder.decode_scalar(1) == 0
        assert DirectionEncoder.decode_scalar(2) == 1

    def test_class_name(self):
        assert DirectionEncoder.class_name(0) == "DOWN"
        assert DirectionEncoder.class_name(1) == "FLAT"
        assert DirectionEncoder.class_name(2) == "UP"

    def test_encode_invalid_labels_raises(self):
        """Labels outside {-1, 0, 1} should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid direction labels"):
            DirectionEncoder.encode(np.array([0, 1, 3]))

    def test_validate_classes(self):
        assert DirectionEncoder.validate_classes(np.array([0, 1, 2])) is True
        assert DirectionEncoder.validate_classes(np.array([0, 1, 3])) is False
        assert DirectionEncoder.validate_classes(np.array([-1, 0, 1])) is False

    def test_validate_labels(self):
        assert DirectionEncoder.validate_labels(np.array([-1, 0, 1])) is True
        assert DirectionEncoder.validate_labels(np.array([0, 1, 2])) is False

    def test_constants_consistency(self):
        """Ensure mapping dicts and constants are in sync."""
        assert DirectionEncoder.LABEL_TO_CLASS[DirectionEncoder.SELL_LABEL] == DirectionEncoder.DOWN_CLASS
        assert DirectionEncoder.LABEL_TO_CLASS[DirectionEncoder.HOLD_LABEL] == DirectionEncoder.FLAT_CLASS
        assert DirectionEncoder.LABEL_TO_CLASS[DirectionEncoder.BUY_LABEL] == DirectionEncoder.UP_CLASS
        assert DirectionEncoder.NUM_CLASSES == 3

    def test_decode_single_int(self):
        """Scalar int input returns scalar int output."""
        result = DirectionEncoder.decode(2)
        assert result == 1
        assert isinstance(result, int)
