# Data protection and security notes

This prototype only uses synthetic data. These notes set out how the same
design would handle real data in a regulated care setting. They are a
starting point for a Data Protection Impact Assessment (DPIA), not legal
advice, and would need sign-off from the provider's data protection lead.

## Why a DPIA is needed

Care notes contain health data (special category data under UK GDPR), and
the retention model profiles employees. Both are high-risk processing, so a
DPIA should be completed before any real data is used.

## What the design already does

| Principle | How CarePulse handles it |
|---|---|
| Data minimisation | Names, phone numbers, NHS numbers, emails and postcodes are removed before any model reads a note. The model only sees events, not identities. |
| Local processing | The LLM runs locally through Ollama, so notes are not sent to an outside AI service. |
| Purpose limitation | Retention scores are for supportive conversations only. The API and dashboard say so on every response. |
| No solely automated decisions | Safeguarding concerns always go to a person. The system cannot close a concern by itself. Retention scores never trigger any action on their own (UK GDPR Article 22). |
| Fairness | Sex and age are not model inputs. They are used only to check whether the model treats groups differently (see results). |
| Access control | API keys map to roles. Only managers can see retention scores or make review decisions. Every refused request is logged. |
| Accountability | Every plan, note check, risk view and review decision is written to a tamper-evident audit log. The log stores hashes of note text, not the text. |
| Security basics | Input length limits, security headers, constant-time key comparison, no secrets in code. |

## What would be added before going live

- A completed DPIA and updated privacy notices for clients and staff.
- Single sign-on (for example Microsoft Entra ID) instead of API keys.
- Encryption at rest for the review database, and TLS everywhere.
- Retention periods for each data type, with automatic deletion.
- Alignment with the NHS Data Security and Protection Toolkit and Cyber Essentials.
- Penetration testing and a review of the redaction rules on real notes.
- A process for staff to see and question their own retention data.
