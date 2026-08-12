from pathlib import Path

from src.dataset_registry import DatasetMetadata, export_registry_json, get_dataset_metadata, list_dataset_ids


def test_dataset_registry_contains_expected_datasets():
    dataset_ids = list_dataset_ids()
    assert "uci_hd_cleveland" in dataset_ids
    assert "uci_hd_hungarian" in dataset_ids
    assert "uci_hd_switzerland" in dataset_ids
    assert "uci_hd_va" in dataset_ids

    metadata = get_dataset_metadata("uci_hd_cleveland")
    assert isinstance(metadata, DatasetMetadata)
    assert metadata.source_file.endswith("processed.cleveland.data")
    assert metadata.target_column == "num"


def test_export_registry_json(tmp_path: Path):
    output_path = tmp_path / "registry.json"
    result = export_registry_json(output_path)

    assert result.exists()
    content = result.read_text()
    assert "uci_hd_cleveland" in content
    assert "UCI Heart Disease Cleveland" in content


def test_registered_datasets_have_collection_id():
    for ds in list_dataset_ids():
        metadata = get_dataset_metadata(ds)
        assert getattr(metadata, "collection_id", None) is not None

    uci_ids = [
        "uci_hd_cleveland",
        "uci_hd_hungarian",
        "uci_hd_switzerland",
        "uci_hd_va",
    ]
    collections = {get_dataset_metadata(ds).collection_id for ds in uci_ids}
    assert len(collections) == 1
