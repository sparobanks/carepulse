# User guide

This guide is for coordinators and registered managers. You do not need any
technical knowledge to use CarePulse.

## Starting it

On the computer it is installed on, run `make serve` and open
http://localhost:8000. Enter the API key you were given at the top right.
For the demo, use `demo-coordinator`, `demo-manager` or `demo-auditor`.

## Visit matching (coordinators and managers)

1. Open **Visit matching** and click **Plan morning shift**.
2. The table compares the suggested plan with a simple "nearest free carer"
   approach. Green numbers are where the suggested plan does better.
3. The map shows each carer's run as a coloured line from their home.
4. To value continuity more (clients seeing familiar faces), raise the
   **continuity weight** and plan again. Travel usually goes up a little.

The plan is a draft. You always approve it before it is sent to carers.
Rules that are never broken: skills the client needs, carer-gender
preferences, the visit's time window and each carer's shift.

## Staff retention (managers only)

**Show highest risk** lists carers who may be thinking of leaving, with the
main reasons and a suggested way to help, such as keeping their rota stable
or talking about training.

Use this to start a supportive conversation. Never use it for disciplinary
action, pay decisions or choosing who to keep.

## Care notes (coordinators and managers)

Paste a visit note and click **Check note**. CarePulse first removes the
person's name and contact details, then looks for:

- **Safeguarding**: possible abuse or neglect
- **Missed tasks**: care that was planned but did not happen
- **Deterioration**: signs someone's health is getting worse
- **Quality risks**: late visits, complaints, faulty equipment, hazards

Each concern shows the exact words from the note that it is based on.

What happens next:

| Outcome | Meaning |
|---|---|
| Urgent review | Possible safeguarding. A manager must look at it. |
| Review | Serious or uncertain. A person should check it. |
| Record | Low-risk issue, logged for follow-up. |
| No action | Nothing found. |

CarePulse does not replace your safeguarding policy. If you are worried
about someone, follow your normal procedure straight away.

## Review queue

Urgent items are shown first. A manager chooses **Confirm concern**,
**Escalate** or **Not a concern**. Each decision is saved in the audit log.
