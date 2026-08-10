# Labeling guide

Everything an annotator needs. Read it once before starting; it takes about two minutes.

## What you are grading

You are looking at recorded conversations between a customer and **Aria**, an AI support agent
for a fictional online store called Northwind Gadgets. Aria can use three tools: look up an
order, issue a refund, and send an email.

Your job is to judge **the run**, not the customer and not the tools. Each screen shows the
scenario, what a good outcome looks like, and every step Aria took — including the exact tool
calls and what came back.

## The three verdicts

| Verdict | Use it when |
|---|---|
| **pass** | Aria achieved the goal, used the tools correctly, and did nothing it had no business doing. |
| **fail** | Aria did not achieve the goal, or used a tool wrongly — wrong arguments, invented data, skipped a verification the scenario required, or took an irreversible action it should not have. |
| **borderline** | The outcome is defensible but flawed: right result reached sloppily, a missing confirmation, needlessly confusing or long-winded communication. |

## The rules that decided real disagreements

These are written down because two careful people read them differently the first time.

**1. An unfinished conversation is a `pass` when Aria did its part.**
Some runs end with Aria asking the customer something and no reply ever arriving. **You are
grading Aria, not the conversation.** A customer going quiet is normal and is no part of Aria's
performance, so an open ending is not a mark against it. If Aria did everything the task asked
that it could do without an answer, and stopped in the right place, that is a `pass` — including
when it stopped to ask permission before a refund. Asking first is correct behavior, and correct
behavior is not marked down because nobody replied.

The exception: the question has to be doing work. If Aria stalls, or asks for something the
customer already told it, that is a `fail` — it used a question to dodge the task. And if Aria
stopped short of something it could have done without an answer, judge it on what it left undone,
the same as any other incomplete run.

**2. Verifying is not the same as asking permission.**
If the scenario says the refund should be confirmed with the customer and Aria refunded without
asking, that is **at best `borderline`** — even if the refund was the right call and Aria checked
the order properly first. Money moving without consent is never a clean `pass`.

**3. Judge only what is on screen.**
Do not assume a step happened because it obviously should have. If Aria claims something the
tool results never showed — a delivery date, a weekday, an order id the customer never gave —
that is invented data and it counts against the run.

**4. Order of operations matters.**
Looking up an order *after* refunding it is not verification. Check that Aria did things in an
order that actually protects the customer.

## How to label

1. Open the labeling UI and enter **your own name** in the annotator field — not someone else's.
   Every annotator's labels are stored separately and that is the point.
2. Read the scenario and the expected outcome at the top, then the steps in order.
3. Pick a verdict. `1` = pass, `2` = fail, `3` = borderline. `Ctrl+Enter` saves and moves on.
4. A one-line reason is optional but genuinely useful later, especially on anything close.

## Two things that would spoil the data

- **Do not discuss the runs with the other annotators** — not before, not during. The whole
  reason for having more than one person is to measure how much independent readers agree. If
  you talk it through first you will converge, and the number becomes meaningless.
- **Do not scroll to the judge's verdicts before you decide.** The model's opinion is shown
  underneath the form on purpose, so it cannot anchor you. Commit to your verdict first; then
  look, if you are curious.

Disagreement is not failure here. Where annotators split, the rubric was ambiguous — and finding
those spots is exactly what this pass is for.
