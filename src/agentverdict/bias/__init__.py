"""Bias probes: measuring where the judge is systematically wrong.

Calibration asks how far the judge agrees with us. The probes in this package ask a
different question — whether the judge's answer depends on things that are not the
run: the order the prompt's sections happen to be in, how many words the agent used
to say the same thing. Each probe re-judges the same trajectories through a
deliberately perturbed prompt and reports the signed movement of the verdict
distribution, with the judge's own re-judge noise as the floor it is measured against.

Two boundaries hold the whole package up. Perturbations are pure functions over
rendered strings, so no probe can write filler into the golden dataset; and probe
verdicts live in their own tables, so a verdict on a deliberately corrupted prompt
can never reach calibration or the merge gate.
"""
