"""Regression gating: is this candidate eval run worse than the baseline, or just different?

The gate is what turns a pile of stored verdicts into a merge decision. It compares
one eval run against a stored baseline run, task by task, and blocks the merge only
when the drop survives resampling the tasks it was measured on.

Everything here is pure analysis over verdicts already in the database: no model is
called and no API key is needed, so the gate costs nothing to run, is reproducible
from a seed, and cannot fail on a rate limit in the middle of a pull request.
"""
