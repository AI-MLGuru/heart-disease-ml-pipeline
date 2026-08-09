from src.dataset_registry import DATASET_REGISTRY, DatasetMetadata, get_dataset_metadata, list_dataset_ids


def test_dataset_registry_contains_expected_datasets():
    dataset_ids = list_dataset_ids()
    assert "uci_hd_cleveland" in dataset_ids
    assert "uci_hd_hungarian" in dataset_ids
    assert "uci_hd_switzerland" in dataset_ids
    assert "uci_hd_va" in dataset_ids

    metadata = get_dataset_metadata("uci_hd_cleveland")
    assert isinstance(metadata, DatasetMetadata)
    assert metadata.source_file == "processed.cleveland.data"
    assert metadata.target_column == "num"
