# Literature Review Workflow

## Required fields for every paper

Record: complete citation, DOI/arXiv/ACL ID, canonical URL, venue/version, model architectures, tasks/datasets, head representation, importance measure, intervention/pruning mechanism, matching/transfer method, metrics, main findings, limitations, code/data availability, full-text verification status, and relevance to RQ1–RQ6.

## Status labels

- `record_verified`: official publisher/proceedings/arXiv metadata checked.
- `abstract_verified`: official abstract checked; methods/results not fully inspected.
- `full_text_reviewed`: full paper inspected and methods/results extracted.
- `citation_chain_reviewed`: backward and forward citation chains checked.
- `needs_verification`: details remain uncertain.

Do not infer methods, authors or exact numerical results from titles alone.

## Citation-chain closure

For every high-overlap paper, inspect:
1. Its references for prior head alignment, pruning, causal attribution and transfer.
2. Forward citations for follow-up methods and replications.
3. Code repositories and appendices for implementation details.
4. Corrections, camera-ready versions and venue changes.

Prioritize HIES, IEICE 2026, TAHA, CAHP, Causal Head Gating, Chiang & Chen, Verma & Elbayad, and Budhraja et al.

## Copyright

Store metadata, notes, lawful links and BibTeX. Do not commit publisher PDFs or copyrighted full-text papers unless the license explicitly permits redistribution.
