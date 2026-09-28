"""Замер локального хранения без интернета и ML.

python -m scripts.benchmark_backend --records 10000
Новая временная база удаляется после замера. tracemalloc не измеряет весь RSS.
"""

import argparse
import json
import tempfile
import time
import tracemalloc
from pathlib import Path

from app.backend.contracts import DocumentRecord, SearchRequest, SourcePage
from app.backend.repository import Repository


def benchmark(count: int) -> dict:
    if not 1 <= count <= 10_000:
        raise ValueError("Число документов должно быть от 1 до 10000")
    with tempfile.TemporaryDirectory(prefix="trendanalyser-benchmark-") as directory:
        repository = Repository(Path(directory) / "benchmark.sqlite3")
        request = SearchRequest(topic="Synthetic benchmark", max_results=count)
        tracemalloc.start()
        timings = []
        for _ in range(2):
            job = repository.create_job(request)
            repository.start_job(job.id)
            started = time.perf_counter()
            for start in range(0, count, 100):
                end = min(start + 100, count)
                docs = tuple(DocumentRecord(
                    source="crossref", source_id=f"10.9999/benchmark-{number}",
                    doi=f"10.9999/benchmark-{number}", title=f"Synthetic research {number}",
                    abstract="Test metadata about technology. " * 20,
                    publication_year=2024, date_precision="year",
                    url=f"https://doi.org/10.9999/benchmark-{number}",
                ) for number in range(start, end))
                repository.ingest_page(job.id, SourcePage(
                    documents=docs, scanned=len(docs), total_available=count, exhausted=end == count,
                ))
            repository.finish_job(job.id, "succeeded")
            timings.append(round(time.perf_counter() - started, 3))
        query_start = time.perf_counter()
        sample = repository.list_documents(query="research 99", limit=20)
        query_seconds = round(time.perf_counter() - query_start, 4)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stored = repository.list_documents(limit=1).total
        if stored != count:
            raise AssertionError("Повторная загрузка изменила число уникальных документов")
        return {
            "synthetic_documents": count,
            "page_size": 100,
            "first_load_seconds": timings[0],
            "repeat_load_seconds": timings[1],
            "unique_documents_after_repeat": stored,
            "literal_search_seconds": query_seconds,
            "search_matches": sample.total,
            "peak_python_allocations_mib": round(peak / 1024 / 1024, 2),
            "limitations": "Synthetic local metadata, no network/ML. Memory excludes native SQLite buffers and OS cache.",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10_000)
    arguments = parser.parse_args()
    print(json.dumps(benchmark(arguments.records), indent=2))
