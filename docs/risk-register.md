# Risk register

| # | Risk | Likelihood | Impact | What reduces it | Owner |
|---|---|---|---|---|---|
| 1 | A real safeguarding concern is missed | Medium | Very high | Rules always run as a safety net; safeguarding threshold tuned for at least 95% recall; every safeguarding flag goes to a person; carers keep the usual reporting route | Registered manager |
| 2 | The LLM invents a concern that is not in the note | Medium | Medium | Evidence must be an exact quote from the note or the finding is dropped; temperature 0; failures send the note to a person | Technical lead |
| 3 | Too many false alarms, so staff ignore the queue | Medium | High | Track the share of notes sent for review; tune thresholds with coordinators; group low-risk findings as "record" | Technical lead |
| 4 | Retention scores used against staff | Low | High | Scores limited to managers; "supportive use only" on every response; reasons framed as actions the provider can take | Registered manager |
| 5 | Model is unfair to a group of staff | Medium | High | Sex and age not used as inputs; per-group checks of accuracy and flag rates in every run; review before each retrain | Technical lead |
| 6 | Real data differs from synthetic data, so results drop | High | Medium | Treat synthetic results as a test of the method only; re-run all evaluations on PBL data before use | Technical lead |
| 7 | Rota data is incomplete or late | High | Medium | Matching skips visits it cannot place and reports them; data quality checks before each run | Coordinator |
| 8 | Personal data leaks through the AI model | Low | Very high | Redaction before any model call; model runs locally; audit log stores hashes, not text | Data protection lead |
| 9 | Optimised rota ignores something a coordinator knows | Medium | Medium | Plans are drafts for a coordinator to approve; weights are visible and adjustable | Coordinator |
| 10 | Audit log is changed after the fact | Low | High | Hash chain; `/audit/verify` shows where the chain breaks | Auditor |
