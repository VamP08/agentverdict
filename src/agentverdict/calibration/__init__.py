"""Calibration: measuring how far an LLM judge can be trusted.

A judge is only useful if its verdicts track human judgment. This package scores a
judge against the human-labeled golden dataset and reports the result with enough
statistics to act on: agreement, chance-corrected agreement, uncertainty, and the
specific cases where judge and humans part ways.
"""
