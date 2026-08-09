from src.dataset_quality import dataset_audit, audit_all_datasets


def test_dataset_audit_reports_target_distribution_and_missing():
    audit = dataset_audit("uci_hd_cleveland")

    assert audit["dataset_id"] == "uci_hd_cleveland"
    assert audit["rows"] == 303
    assert audit["columns"] == 17 or audit["columns"] == 14
    assert "target_distribution" in audit
    assert "missing_values" in audit


def test_audit_all_datasets_includes_all_registry_entries():
    all_audits = audit_all_datasets()
    assert set(all_audits.keys()) == {
        "uci_hd_cleveland",
        "uci_hd_hungarian",
        "uci_hd_switzerland",
        "uci_hd_va",
    }
    assert all_audits["uci_hd_hungarian"]["rows"] == 294
