"""The checks every correction must pass (`correct`): placeholders, length, edit size,
language, 〈?〉 gap fill, and the re-score."""

from ocrfix.verify.checks import CHECKS, Check

__all__ = ["CHECKS", "Check"]
