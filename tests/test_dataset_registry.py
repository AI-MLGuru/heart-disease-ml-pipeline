from pathlib import Path

from src.dataset_registry import (
    DatasetMetadata,
    export_registry_json,
    get_dataset_metadata,
    list_dataset_ids,
)


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


def test_uci_site_and_geography():
    clev = get_dataset_metadata("uci_hd_cleveland")
    va = get_dataset_metadata("uci_hd_va")
    hun = get_dataset_metadata("uci_hd_hungarian")
    ch = get_dataset_metadata("uci_hd_switzerland")

    # site fields present
    assert clev.site is not None
    assert va.site is not None
    assert hun.site is not None
    assert ch.site is not None

    # countries and regions as expected
    assert clev.country == "USA"
    assert va.country == "USA"
    assert clev.region == "North America"
    assert va.region == "North America"

    assert hun.country == "Hungary"
    assert hun.region == "Europe"
    assert ch.country == "Switzerland"
    assert ch.region == "Europe"


def test_pending_dataset_excluded_from_acquired_list():
    # all registered datasets should include the pending Nigerian candidate
    all_ids = list_dataset_ids()
    assert "ng_kano_cad_506" in all_ids

    # acquired-only listing should not include the pending dataset
    from src.dataset_registry import list_dataset_ids_by_status

    acquired = list_dataset_ids_by_status("ACQUIRED")
    assert "ng_kano_cad_506" not in acquired
    # ensure the four UCI datasets are still acquired
    for ds in [
        "uci_hd_cleveland",
        "uci_hd_hungarian",
        "uci_hd_switzerland",
        "uci_hd_va",
    ]:
        assert ds in acquired
