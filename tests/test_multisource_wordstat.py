"""Wordstat CSV facts remain separate from derived search conclusions."""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from app.pilot.multisource.imports import read_csv_document
from app.pilot.multisource.queries import build_manual_profile
from app.pilot.multisource.contracts import QueryTerm, SearchObservation, SourceSnapshot, WordstatImportReceipt
from app.pilot.multisource.store import SignalStore
from app.pilot.multisource.wordstat import DynamicsMapping, TopMapping, import_wordstat_csv, parse_dynamics, parse_top
from app.runtime.jobs import TaskFailure


NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)
HASH = "a" * 64


def _document(tmp_path: Path, content: str, *, encoding: str = "utf-8-sig"):
    path = tmp_path / "source.csv"
    path.write_bytes(content.encode(encoding))
    return read_csv_document(path, encoding=encoding, delimiter=";")


def _profile():
    return build_manual_profile("молекулярная память", "Запись данных в молекулы",
                                seed_terms=("ДНК память",), primary_phrase="ДНК память", confirmed_at=NOW)


def _mapping(**changes):
    values = dict(date_column="Месяц", count_column="Запросов", share_column="Доля",
                  date_format="MM.YYYY", share_unit="percent", phrase="ДНК память")
    values.update(changes)
    return DynamicsMapping(**values)


def test_monthly_counts_percentage_and_partial_current_month(tmp_path: Path) -> None:
    document = _document(tmp_path, "Месяц;Запросов;Доля\n08.2026;1 234;0,25%\n09.2026;45;0,01%\n")
    parsed = parse_dynamics(document, _mapping(), _profile(), snapshot_hash=HASH,
                            observed_at=NOW, available_at=NOW)
    assert parsed.rejected == ()
    august, september = parsed.observations
    assert august.count == 1234 and august.share_raw == "0.25" and august.share_fraction == "0.0025"
    assert august.period_start == date(2026, 8, 1) and august.period_end == date(2026, 8, 31)
    assert august.is_complete_period is True
    assert september.period_end == date(2026, 9, 19) and september.is_complete_period is False


def test_observed_zero_missing_and_quantized_share_are_distinct(tmp_path: Path) -> None:
    document = _document(tmp_path, "Месяц;Запросов;Доля\n05.2026;0;0\n06.2026;;\n07.2026;3;0,00%\n")
    parsed = parse_dynamics(document, _mapping(), _profile(), snapshot_hash=HASH,
                            observed_at=NOW, available_at=NOW)
    first, second, third = parsed.observations
    assert first.value_status == "observed" and first.count == 0 and first.share_fraction == "0"
    assert second.value_status == "missing" and second.count is None
    assert third.count == 3 and third.normalization_status == "quantized_zero" and third.share_fraction is None


def test_unknown_share_unit_is_never_assumed_a_fraction(tmp_path: Path) -> None:
    document = _document(tmp_path, "Месяц;Запросов;Доля\n08.2026;100;0,25\n")
    parsed = parse_dynamics(document, _mapping(share_unit="unknown"), _profile(), snapshot_hash=HASH,
                            observed_at=NOW, available_at=NOW)
    item = parsed.observations[0]
    assert item.count == 100 and item.share_raw == "0.25" and item.share_fraction is None
    assert item.normalization_status == "unknown_unit"


def test_duplicate_month_and_unconfirmed_phrase_fail_closed(tmp_path: Path) -> None:
    document = _document(tmp_path, "Месяц;Запросов;Доля\n08.2026;1;0,1%\n08.2026;2;0,2%\n")
    with pytest.raises(TaskFailure, match="повторяется"):
        parse_dynamics(document, _mapping(), _profile(), snapshot_hash=HASH,
                       observed_at=NOW, available_at=NOW)
    with pytest.raises(TaskFailure):
        parse_dynamics(document, _mapping(phrase="RAG"), _profile(), snapshot_hash=HASH,
                       observed_at=NOW, available_at=NOW)


def test_invalid_unit_and_count_are_reported_as_rejections(tmp_path: Path) -> None:
    document = _document(tmp_path, "Месяц;Запросов;Доля\n07.2026;1;0,1%\n08.2026;1 2;0,2%\n")
    parsed = parse_dynamics(document, _mapping(), _profile(), snapshot_hash=HASH,
                            observed_at=NOW, available_at=NOW)
    assert len(parsed.observations) == 1 and parsed.rejected[0].line == 3
    wrong = parse_dynamics(document, _mapping(share_unit="fraction"), _profile(), snapshot_hash=HASH,
                           observed_at=NOW, available_at=NOW)
    assert wrong.observations == () and len(wrong.rejected) == 2


def test_top_rows_only_propose_terms_and_never_claim_growth(tmp_path: Path) -> None:
    document = _document(tmp_path, "Фраза;Частота\nДНК память;100\nДНК архив;50\nДНК память;100\n")
    parsed = parse_top(document, TopMapping("Фраза", "Частота"), snapshot_hash=HASH)
    assert tuple(item.text for item in parsed.terms) == ("ДНК память", "ДНК архив")
    assert all(item.status == "proposed" and item.origin == "wordstat" for item in parsed.terms)
    assert parsed.rejected[0].reason == "duplicate_phrase"


def test_import_persists_verified_snapshot_observations_and_receipt(tmp_path: Path) -> None:
    store = SignalStore(tmp_path / "app")
    profile_hash = store.put_object(_profile())
    path = tmp_path / "export.csv"
    path.write_bytes("Месяц;Запросов;Доля\n07.2026;100;0,10%\n08.2026;120;0,12%\n".encode("utf-8-sig"))
    with pytest.raises(TaskFailure):
        import_wordstat_csv(store, path, profile_hash, kind="dynamics",
                            mapping=_mapping(expected_from=date(2026, 7, 1), expected_to=date(2026, 8, 1)),
                            encoding="utf-8-sig", delimiter=";", retention="unknown", observed_at=NOW)
    receipt_hash = import_wordstat_csv(store, path, profile_hash, kind="dynamics",
                                       mapping=_mapping(expected_from=date(2026, 7, 1), expected_to=date(2026, 8, 1)),
                                       encoding="utf-8-sig", delimiter=";", retention="local_allowed", observed_at=NOW)
    assert receipt_hash == import_wordstat_csv(
        store, path, profile_hash, kind="dynamics",
        mapping=_mapping(expected_from=date(2026, 7, 1), expected_to=date(2026, 8, 1)),
        encoding="utf-8-sig", delimiter=";", retention="local_allowed", observed_at=NOW)
    receipt = store.get_object(receipt_hash, WordstatImportReceipt)
    snapshot = store.get_object(receipt.snapshot_hash, SourceSnapshot)
    assert receipt.row_count == receipt.accepted_count == 2 and receipt.rejected_count == 0
    assert snapshot.coverage == "complete" and snapshot.comparable is True
    observations = tuple(store.get_object(item, SearchObservation) for item in receipt.observation_hashes)
    assert [item.count for item in observations] == [100, 120]
    assert all(item.snapshot_hash == receipt.snapshot_hash and item.query_profile_hash == profile_hash
               for item in observations)
    assert store.verify_raw(receipt.raw_hash, "csv").read_bytes() == path.read_bytes()
    assert not (store.root / "catalogue.json").exists()


def test_partial_import_does_not_claim_comparability(tmp_path: Path) -> None:
    store = SignalStore(tmp_path / "app")
    profile_hash = store.put_object(_profile())
    path = tmp_path / "export.csv"
    path.write_text("Месяц;Запросов;Доля\n07.2026;100;0,10%\n08.2026;bad;0,12%\n", encoding="utf-8")
    receipt_hash = import_wordstat_csv(store, path, profile_hash, kind="dynamics",
                                       mapping=_mapping(expected_from=date(2026, 7, 1), expected_to=date(2026, 8, 1)),
                                       encoding="utf-8-sig", delimiter=";", retention="local_allowed", observed_at=NOW)
    receipt = store.get_object(receipt_hash, WordstatImportReceipt)
    snapshot = store.get_object(receipt.snapshot_hash, SourceSnapshot)
    assert (receipt.accepted_count, receipt.rejected_count, receipt.unselected_count) == (1, 1, 0)
    assert snapshot.coverage == "partial" and snapshot.comparable is False


def test_transfer_rights_are_explicit_and_immutable(tmp_path: Path) -> None:
    store = SignalStore(tmp_path / "app")
    profile_hash = store.put_object(_profile())
    path = tmp_path / "export.csv"
    path.write_text("Месяц;Запросов;Доля\n08.2026;100;0,10%\n", encoding="utf-8")
    options = dict(kind="dynamics", mapping=_mapping(), encoding="utf-8-sig",
                   delimiter=";", retention="local_allowed", observed_at=NOW)
    local = import_wordstat_csv(store, path, profile_hash, **options)
    with pytest.raises(ValueError, match="rights reference"):
        import_wordstat_csv(store, path, profile_hash, **options, export_right="share_allowed")
    shared = import_wordstat_csv(store, path, profile_hash, **options,
                                 export_right="share_allowed", license_ref="own dataset, 2026-09-19")
    assert shared != local
    assert store.get_object(store.get_object(local, WordstatImportReceipt).snapshot_hash,
                            SourceSnapshot).export_right == "local_only"
    snapshot = store.get_object(store.get_object(shared, WordstatImportReceipt).snapshot_hash, SourceSnapshot)
    assert snapshot.export_right == "share_allowed" and snapshot.license_ref == "own dataset, 2026-09-19"


def test_top_import_orders_by_frequency_without_creating_a_growth_series(tmp_path: Path) -> None:
    store = SignalStore(tmp_path / "app")
    profile_hash = store.put_object(_profile())
    path = tmp_path / "top.csv"
    path.write_text("Фраза;Частота\nДНК архив;5\nДНК носитель;20\n", encoding="utf-8")
    receipt_hash = import_wordstat_csv(store, path, profile_hash, kind="top",
                                       mapping=TopMapping("Фраза", "Частота"), encoding="utf-8-sig",
                                       delimiter=";", retention="local_allowed", observed_at=NOW)
    receipt = store.get_object(receipt_hash, WordstatImportReceipt)
    assert receipt.observation_hashes == () and receipt.top_counts == (20, 5)
    assert [store.get_object(item, QueryTerm).text for item in receipt.term_hashes] == ["ДНК носитель", "ДНК архив"]
    assert store.get_object(receipt.snapshot_hash, SourceSnapshot).comparable is False
