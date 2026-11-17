ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_required_research_documents_exist() -> None:
    required = [
        "research/research_proposal.md",
        "research/research_questions.md",
        "research/novelty_audit.md",
        "research/research_protocol.md",
        "research/methodology/mathematical_definitions.md",
        "research/literature/paper_inventory.csv",
        "configs/feasibility.yaml",
        "configs/main_experiment.yaml",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    assert not missing, f"Missing research foundation files: {missing}"


def test_proposal_contains_primary_research_question() -> None:
    proposal = (ROOT / "research/research_proposal.md").read_text(encoding="utf-8")
    assert "task-conditioned causal correspondence" in proposal.lower()


def test_feasibility_config_declares_pilot_dimensions() -> None:
    import yaml

    config = yaml.safe_load((ROOT / "configs/feasibility.yaml").read_text(encoding="utf-8"))
    assert config["data"]["sample_count"] == 256
    assert config["extraction"]["expected_heads"] == 144
    assert config["bootstrap"]["pilot_replicates"] >= 50
    assert config["bootstrap"]["permutation_replicates"] >= 20


def test_experiment_modules_import_without_model_extras() -> None:
    __import__("cra_hp.extraction")
    __import__("cra_hp.experiments.representation_pilot")
