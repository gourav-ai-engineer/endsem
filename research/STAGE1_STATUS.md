# Stage 1 Status

**Status:** Foundation files committed; review remains open.

## Delivered

- Research proposal, research questions and falsifiable hypotheses.
- Provisional novelty audit with explicit caveats.
- Literature inventory CSV and BibTeX bibliography.
- Mathematical definitions for CKA, partial matching, bootstrap confidence, causal response, importance transfer and pruning utility.
- Staged experiment protocol and go/no-go gates.
- Feasibility and main experiment YAML configurations.
- Reproducibility/data-governance guidance.
- Python package metadata, setup guide, initial smoke tests and GitHub Actions workflow.

## Literature verification notes

- HIES: official ACL Anthology metadata, author list, abstract and PDF URL verified. The matrix still marks detailed full-text section/table extraction as pending; do not treat abstract-level metrics as a complete method audit.
- IEICE dataset-universality letter: J-STAGE record, six authors, DOI, free-access PDF availability and abstract verified. The full PDF is available from the publisher, but detailed methods/results extraction remains pending.
- Citation-chain review and full-text review of other related papers remain open.
- TAHA is a close transferable-pruning precedent and requires full methods/results extraction before final novelty claims.

## Execution status

- GitHub Actions foundation smoke tests: passed for commit `1f22916d208f27cc47d07c26dd3fc695e403deb7`.
- The later literature metadata-only updates are in this branch after that tested commit; the workflow will rerun on the current head.
- No BERT/GPT-2 model was loaded. No CKA, matching or causal experiments have been executed.
- This stage is not a claim that the research hypothesis has been validated.

## Next stage

Implement Stage 2 extraction and representation analysis as a separate, tested module. Preserve all run manifests and raw metrics. Use the predeclared protocol and do not tune matching thresholds on final test outcomes.
