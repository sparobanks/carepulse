# Model cards

Numbers for every model are in [reports/RESULTS.md](../reports/RESULTS.md).
All were measured on synthetic data, so they show the method works, not how
well it will do on real records.

## Retention risk model

| | |
|---|---|
| Purpose | Estimate the chance a carer leaves in the next 90 days, so a manager can offer support early |
| Not for | Disciplinary action, pay, selection, or any decision about a person made without a conversation |
| Model | LightGBM, calibrated with isotonic regression; logistic regression kept as a simple baseline |
| Inputs | Tenure, weekly hours, overtime, short-notice rota changes, rota volatility, unpaid travel share, split shifts, weekend work, client continuity, sickness days, months since last progression, overdue supervision |
| Not inputs | Sex and age band (kept only to check fairness), names, any health information |
| Training | Monthly snapshots; trained on months 0 to 11, calibrated on 14 to 15, tested on 18 to 21. Two-month gaps stop the 3-month label from leaking into the next split |
| Explanations | SHAP values per person. A reason is only shown if the person's value is on the risky side of the typical value, and each reason comes with a supportive action |
| Fairness check | Accuracy, flag rate and share of leavers caught, by sex and age band, at the top-10% flag line |
| Known limits | Synthetic data with a known cause structure; real staff leave for reasons not in rota data (family, pay elsewhere, health). Expect lower accuracy on real data. Re-check fairness on every retrain |

## Care-notes detectors

| | |
|---|---|
| Purpose | Find possible safeguarding concerns, missed tasks, signs of deterioration and quality risks in free-text visit notes |
| Not for | Replacing a person's judgement or the provider's safeguarding procedure |
| Default | Negation-aware rules (always on), plus an optional local LLM through Ollama |
| Experiment | TF-IDF + logistic regression classifier, to test whether a small learned model generalises to new phrasings |
| Inputs | Note text after redaction of names, phone numbers, NHS numbers, emails and postcodes |
| Outputs | Category, severity, an exact quote from the note as evidence, confidence, and a route |
| Safety design | Safeguarding always goes to urgent review; LLM findings without a matching quote are dropped; if any model fails, the note goes to a person |
| Test design | Notes are built from sentence "families". Test notes use families never seen in training, so a model cannot pass by memorising templates |
| Known limits | Rules were written with knowledge of the generator, so their scores are optimistic. The learned classifier shows how badly a word-based model copes with new phrasings, which is the case for the LLM. Real validation needs real notes labelled by experienced coordinators |
