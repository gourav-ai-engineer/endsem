# Provisional Novelty Audit

**Status:** provisional; not a claim of exhaustive novelty. Last curated: 2026-11-19.

## Proposed narrow gap

The candidate gap is not attention-head similarity, head pruning, transferability, task-specific importance, or head interactions individually. Each has substantial prior art. The narrower proposed question is whether *task-conditioned causal-response correspondence between bidirectional encoder and causal decoder heads* can transfer pruning utility, with calibrated abstention under distribution shift.

## Closely related work and implications

| Work | Relevant idea | Overlap | Remaining distinction to test |
|---|---|---|---|
| Michel et al. (2019), Are Sixteen Heads Really Better than One? | Head ablation and pruning across tasks | Establishes head redundancy and importance analysis | Does not establish BERT-to-GPT-2 causal utility transfer |
| Voita et al. (2019), Analyzing Multi-Head Self-Attention | Specialized heads and pruning | Head specialization/prunability | Not cross-architecture causal-response transfer |
| Budhraja et al. (2020), On the Weak Link Between Importance and Prunability | Importance scores may poorly predict prunability | Direct warning against equating importance with utility | Motivates explicit target ablation validation |
| Zhang et al. (2021), Single-Shot Meta-Pruning for Attention Heads | Meta-pruning and transfer | Transfer of pruning decisions | Must distinguish proposed architecture-pair and causal signature |
| Chiang & Chen (2021), Why Can You Lay Off Heads? | Head importance coherence/transfer across models/tasks | Transfer-related head importance | Not the same encoder-decoder causal correspondence question |
| Li et al. (2021), Differentiable Subset Pruning of Transformer Heads | Learned subset selection | Pruning optimization baseline | Not cross-architecture correspondence |
| Bansal et al. (2023), Rethinking the Role of Scale for In-Context Learning | Task-specific head importance in large decoder LMs | Decoder head importance and cross-task generalization | Not encoder-to-decoder matching |
| Shapley Head Pruning (2022) | Interaction-aware head importance | Head interactions | Does not by itself establish cross-architecture transfer |
| Wu et al. (ACL 2020), Similarity Analysis of Contextual Word Representation Models | Similarity of contextual representations across models | Cross-model representation similarity | Similarity is not causal utility transfer |
| Dar et al. (ACL 2023), Analyzing Transformers in Embedding Space | Transformer parameter/representation alignment | Alignment concepts | Not head-level pruning-utility transfer |
| Verma & Elbayad (TMLR 2024), Merging Text Transformer Models from Different Initializations | Feature/head assignment using correlations | Explicit head alignment | Model merging between similar text transformers, not BERT-to-GPT-2 pruning utility |
| ALPS (Findings ACL 2025) | Attention localization and selective adaptation | Head/attention selection | Primarily adaptation/update selection, not equivalent to physical cross-architecture pruning |
| Causal Head Gating (NeurIPS 2025) | Causal gates and head interactions | Causal intervention mechanism | Soft gating/analysis is not the same as cross-architecture transfer and structural pruning |
| TAHA (Frontiers in Plant Science, 2026) | Transferable attention-head alignment/pruning across plant domains in ViTs | Strong prior art for transferable head pruning | Vision Transformer/domain setting; proposed study is encoder-decoder language-model causal utility transfer |
| CAHP (IJCNN 2026) | Complementary, graph-based head pruning | Interaction-aware selection | Encoder-focused pruning; generative cross-architecture transfer remains distinct |
| HIES (Findings EMNLP 2026) | Importance + entropy for stable head ranking | Strong target-model ranking baseline | Not itself cross-architecture causal correspondence |
| IEICE (2026), Examining the Universality Assumption of Attention Head Pruning across Datasets | Dataset-specific pruning choices | Supports distribution-shift evaluation | Does not establish the proposed transfer mechanism |
| Sheth et al. (2026), Pruning via Causal Attribution Preserves Reasoning Performance in Large Language Models (CAP) | Uses per-head masking effects on calibration examples to guide within-model weight pruning | Direct overlap in causal head scoring and pruning motivation | It does not align heads across architectures or test transfer of rankings from an encoder to a decoder |
| Mishra (2026), The Fragility of Reasoning-Importance Attribution for Pruning Language Models | Reports measurement-target and granularity sensitivity, including noisy unit-level rankings and group interaction effects | Strong warning for causal score reliability and single-head-to-group extrapolation | Reasoning-model setting; does not examine cross-architecture head correspondence |

## Selected references

1. Michel, P., Levy, O., & Neubig, G. (2019). *Are Sixteen Heads Really Better than One?* NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2019/file/2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf
2. Voita, E. et al. (2019). *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL. https://aclanthology.org/P19-1580/
3. Budhraja, A. et al. (2020). *On the Weak Link Between Importance and Prunability of Attention Heads.* EMNLP. https://aclanthology.org/2020.emnlp-main.260/
4. Zhang, C. et al. (2021). *Know What You Don't Need: Single-Shot Meta-Pruning for Attention Heads.* arXiv:2011.03770. https://arxiv.org/abs/2011.03770
5. Chiang, C.-H. & Chen, Y.-N. (2021). *Why Can You Lay Off Heads? Investigating How BERT Heads Transfer.* arXiv:2106.07137. https://arxiv.org/abs/2106.07137
6. Li, J. et al. (2021). *Differentiable Subset Pruning of Transformer Heads.* TACL. https://aclanthology.org/2021.tacl-1.86/
7. Ma, X. et al. (2021). *Contributions of Transformer Attention Heads in Multi- and Cross-lingual Tasks.* ACL. https://aclanthology.org/2021.acl-long.152/
8. Bansal, H. et al. (2023). *Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale.* ACL. https://aclanthology.org/2023.acl-long.660/
9. Wu, Z. et al. (2020). *Similarity Analysis of Contextual Word Representation Models.* ACL. https://aclanthology.org/2020.acl-main.422/
10. Dar, G. et al. (2023). *Analyzing Transformers in Embedding Space.* ACL. https://aclanthology.org/2023.acl-long.893/
11. Verma, V. & Elbayad, M. (2024). *Merging Text Transformer Models from Different Initializations.* TMLR/OpenReview. https://openreview.net/forum?id=nWnYSLncXa
12. Chen, Y. et al. (2025). *Attention Localization and Pruning Strategy for Efficient Adaptation of Large Language Models (ALPS).* Findings ACL 2025. https://aclanthology.org/2025.findings-acl.612/
13. Nam, J. et al. (2025). *Causal Head Gating.* arXiv:2505.13737. https://arxiv.org/abs/2505.13737
14. Livertovsky, Y. et al. (2026). *Complementary Attention Head Pruning for Efficient Transformers (CAHP).* IJCNN 2026. https://doi.org/10.1109/IJCNN69854.2026.11728979
15. Choi, S. et al. (2026). *Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning (HIES).* Findings EMNLP 2026. https://aclanthology.org/2026.findings-emnlp.2191/
16. *Examining the Universality Assumption of Attention Head Pruning across Datasets.* IEICE Transactions on Information and Systems (2026). DOI: 10.1587/transinf.2026PAL0001.
17. *Lightweight Plant Phenotypic Feature Extraction via Transferable Attention Head Pruning in Vision Transformers* (TAHA). Frontiers in Plant Science (2026). https://doi.org/10.3389/fpls.2026.1747598
18. *Shapley Head Pruning.* arXiv:2210.05709. https://arxiv.org/abs/2210.05709
19. Sheth, A. et al. (2026). *Pruning via Causal Attribution Preserves Reasoning Performance in Large Language Models.* arXiv:2606.19350. https://arxiv.org/abs/2606.19350
20. Mishra, H. (2026). *The Fragility of Reasoning-Importance Attribution for Pruning Language Models.* arXiv:2611.22234. https://arxiv.org/abs/2611.22234

## Verification status and required closure

- Bibliographic records and abstracts have been checked for the key 2026 works; the HIES full paper and CAHP full paper were previously inspected.
- CAP's arXiv full text and abstract were checked on 2026-11-19; it performs within-model causal head scoring and weight pruning, so the remaining distinction here is cross-architecture transfer rather than causal scoring itself.
- Mishra's arXiv abstract was checked on 2026-11-19. The reported granularity and grouping concerns are directly relevant; its full methods and results still need review before detailed comparison.
- The IEICE full text has not yet been fully extracted for methods, datasets, limitations and result tables. Treat its row as abstract/record verified only.
- TAHA is a close 2026 precedent and its full methods/results must be read before finalizing the novelty statement.
- Citation chaining is not complete. For every directly relevant paper, inspect its references and forward citations for head alignment, causal intervention, pruning transfer, cross-architecture model alignment and domain-shift evaluation.
- Check preprints and proceedings for version changes, corrections and duplicate records.
- Add a verification date, source URL/DOI, full-text status and relevance notes to the machine-readable literature inventory.

## Safe novelty wording

Until the closure work is complete, use: "We investigate whether task-conditioned causal-response correspondence can support pruning-utility transfer between a bidirectional encoder and a causal decoder."

Do not write "the first method", "no prior work", or "novel cross-architecture head pruning" without a documented systematic search and direct-equivalence check.
