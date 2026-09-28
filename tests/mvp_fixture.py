"""Synthetic independent studies, with fixed technology vocabularies and dates."""

from datetime import date

from app.backend.contracts import DocumentRecord, SourcePage


VOCABULARIES = [
    ("Silicon microring resonator photonic neural networks", "silicon microring resonance wavelength cavity waveguide"),
    ("Organic photoelectric synaptic transistors for neural computing", "organic transistor polymer synapse plasticity charge"),
    ("Optical diffractive neural networks with metasurfaces", "diffractive metasurface diffraction propagation layers hologram"),
]


def research_groups(result):
    """All displayed research groups, preserving score order across UI sections."""
    from app.ml.selection import ranking_key
    return sorted([c for key in ("candidates", "preliminary_signals", "established")
                   for c in result.get(key, [])], key=ranking_key)


def records(year, source="openalex"):
    count = max(3, (year - 2018) * 2)
    for topic, (title, vocabulary) in enumerate(VOCABULARIES):
        for number in range(count):
            identifier = f"{year}-{topic}-{number}"
            yield DocumentRecord(source=source, source_id=identifier, doi=f"10.9999/mvp-{identifier}",
                title=f"{title}: experimental device {number}",
                abstract=f"{title} require efficient processing. Existing devices face a bandwidth bottleneck. "
                         f"Here we demonstrate an optical neural device using {vocabulary}. "
                         f"The {vocabulary} architecture improves energy efficiency and reduces latency. "
                         f"We implement neural computation with {vocabulary} and evaluate several configurations. "
                         "Further experiments are needed to establish scalability.",
                authors=(f"Researcher {identifier}",), publication_year=year, publication_date=date(year, 6, 1),
                date_precision="day", document_type="article", language="en",
                url=f"https://example.org/study/{identifier}")


class Provider:
    def iter_pages(self, request, cancel):
        documents = tuple(records(request.from_date.year, request.source))
        yield SourcePage(documents=documents, scanned=len(documents), total_available=len(documents), exhausted=True)

    def close(self):
        pass


def snapshot():
    periods, batches = [], []
    for year in range(2020, 2026):
        docs = [{"document_key": d.document_key, "revision_id": d.source_id,
                 "document": d.model_dump(mode="json")} for d in records(year)]
        request = {"topic": "photonic neuromorphic computing", "source": "openalex",
                   "from_date": f"{year}-01-01", "until_date": f"{year}-12-31"}
        job = {"id": str(year), "request": request, "state": "succeeded", "stored": len(docs),
               "scanned": len(docs), "total_available": len(docs), "source_exhausted": True, "skipped": 0}
        periods.append({"id": str(year), "state": "complete", "job": job, "source": "openalex",
                        "from_date": request["from_date"], "until_date": request["until_date"]})
        batches.append({"job_id": str(year), "total": len(docs), "documents": docs})
    return {"schema_version": 1, "history": {"id": "synthetic-history", "state": "succeeded", "contract_version": 2,
            "request": {"topic": "photonic neuromorphic computing", "sources": ["openalex"]}, "periods": periods},
            "batches": batches}


# Ground truth is used only by tests; the model receives publication text and dates.
GROWTH_PROFILES = {
    "new_signal": (0, 0, 0, 0, 4, 16),
    "stable_background": (12, 12, 12, 12, 12, 12),
    "other_background": (8, 8, 8, 8, 8, 8),
    "single_spike": (0, 0, 0, 0, 0, 12),
}


def growth_snapshot():
    """A planted two-year signal among established topics and a one-year burst."""
    vocabularies = {
        "new_signal": ("Wavelength multiplexed microring photonic neural accelerators",
                       "microring resonator wavelength cavity silicon waveguide"),
        "stable_background": VOCABULARIES[1],
        "other_background": VOCABULARIES[2],
        "single_spike": ("Excitable laser photonic spiking neural systems",
                         "laser spiking pulse excitation threshold temporal dynamics"),
    }
    data = snapshot()
    for index, year in enumerate(range(2020, 2026)):
        documents = []
        for kind, profile in GROWTH_PROFILES.items():
            title, vocabulary = vocabularies[kind]
            for number in range(profile[index]):
                identifier = f"growth-{kind}-{year}-{number}"
                record = DocumentRecord(
                    source="openalex", source_id=identifier, doi=f"10.9999/{identifier}",
                    title=f"{title}: experiment {year}-{number}",
                    abstract=f"{title} provide a platform for neural computation. "
                             "Existing devices face a bandwidth bottleneck. "
                             f"Here we demonstrate an optical neural device using {vocabulary}. "
                             f"The {vocabulary} architecture improves energy efficiency and reduces latency. "
                             f"We implement neural computation with {vocabulary} and evaluate several configurations.",
                    authors=(f"Researcher {identifier}",), publication_year=year,
                    publication_date=date(year, 6, 1), date_precision="day", document_type="article",
                    language="en", url=f"https://example.org/study/{identifier}")
                documents.append({"document_key": record.document_key, "revision_id": identifier,
                                  "document": record.model_dump(mode="json")})
        data["batches"][index].update(documents=documents, total=len(documents))
        data["history"]["periods"][index]["job"].update(
            stored=len(documents), scanned=len(documents), total_available=len(documents))
    return data
