"""Local profiles translate only declared aliases and never guess a corpus topic."""

import hashlib
import json
import re
import socket

import pytest

from app.ml import directions


@pytest.mark.parametrize("value,expected", [
    ("фотонная нейроморфика", "photonic neuromorphic computing"),
    ("  ФОТОННЫЕ   НЕЙРОСЕТИ ", "photonic neuromorphic computing"),
    ("Фотонные нейроморфные вычисления", "photonic neuromorphic computing"),
    ("квантовые резервуарные вычисления", "quantum reservoir computing"),
    ("quantum reservoir computation", "quantum reservoir computing"),
    ("хранение данных в ДНК", "dna data storage"),
    ("dna information storage", "dna data storage"),
    ("ИИ", "artificial intelligence"),
    ("искусственный интеллект", "artificial intelligence"),
    ("технологии в ии", "artificial intelligence"),
    ("квантовые вычисления", "quantum computing"),
])
def test_declared_aliases_resolve_without_network(value, expected, monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("No network in profiles"))
    assert directions.resolve_direction(value) == expected


@pytest.mark.parametrize("value", ["materials discovery", "Неизвестная русская технология", "квантовые сенсоры"])
def test_unknown_directions_are_preserved_and_have_no_claimed_profile(value):
    assert directions.resolve_direction(value) == value
    assert directions.direction_profile(value) is None
    assert directions.profile_fingerprint(value) is None


def test_general_quantum_and_ai_are_not_claimed_as_validated_profiles():
    for query in ("квантовые вычисления", "quantum computing", "ИИ", "artificial intelligence"):
        assert directions.direction_profile(query) is None
    assert directions.resolve_direction("квантовые вычисления") != directions.resolve_direction("квантовые резервуарные вычисления")


def test_profile_copies_do_not_allow_callers_to_mutate_resolution_or_fingerprint():
    first = directions.direction_profile("фотонная нейроморфика")
    old_hash = directions.profile_fingerprint("фотонная нейроморфика")
    first["aliases"].append("медицина")
    first["axes"]["carrier"].clear()
    assert directions.direction_profile("медицина") is None
    assert directions.direction_profile("фотонная нейроморфика")["axes"]["carrier"]
    assert directions.profile_fingerprint("фотонная нейроморфика") == old_hash


@pytest.mark.parametrize("query,positive,negative", [
    ("photonic neuromorphic computing", "Photonic neural inference", "Optical data transmission"),
    ("quantum reservoir computing", "Quantum reservoir computing on qubits", "Quantum materials for battery storage"),
    ("dna data storage", "DNA digital data storage", "DNA methylation in biological organisms"),
])
def test_axis_profiles_have_paired_positive_and_negative_examples(query, positive, negative):
    axes = directions.direction_profile(query)["axes"]
    def covered(text):
        return all(any(re.search(pattern, text, re.I) for pattern in patterns) for patterns in axes.values())
    assert covered(positive)
    assert not covered(negative)


def test_fingerprint_matches_effective_config_and_all_aliases_share_it():
    profile = directions.direction_profile("хранение данных в ДНК")
    encoded = json.dumps(profile, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    expected = hashlib.sha256(encoded).hexdigest()
    assert directions.profile_fingerprint("хранение данных в ДНК") == expected
    assert directions.profile_fingerprint("dna data storage") == expected
    assert expected != directions.profile_fingerprint("quantum reservoir computing")


@pytest.mark.parametrize("mutation", ["alias_collision", "empty_axis", "missing_axes", "invalid_regex", "duplicate_id"])
def test_invalid_profile_configuration_fails_explicitly(tmp_path, monkeypatch, mutation):
    first = directions.direction_profile("dna data storage")
    second = directions.direction_profile("quantum reservoir computing")
    if mutation == "alias_collision":
        second["aliases"].append(first["canonical_query"])
    elif mutation == "empty_axis":
        second["axes"]["compute"] = []
    elif mutation == "missing_axes":
        second["axes"] = {}
    elif mutation == "invalid_regex":
        second["axes"]["carrier"] = ["["]
    else:
        second["id"] = first["id"]
    for number, profile in enumerate((first, second)):
        (tmp_path / f"{number}.json").write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(directions, "PROFILE_DIRECTORY", tmp_path)
    directions._profiles.cache_clear()
    try:
        with pytest.raises((ValueError, re.error)):
            directions.direction_profile("dna data storage")
    finally:
        directions._profiles.cache_clear()


@pytest.mark.parametrize("title,parts,expected", [
    ("Quantum reservoir computing with repeated measurements on superconducting devices", [], True),
    ("Reservoir Computing via Quantum Recurrent Neural Networks", [], True),
    ("Parallel Time-Delay Reservoir Computing With Quantum Dot Lasers",
     ["This work theoretically demonstrates a parallel TDRC based on a Fabry-Perot QD laser with multiple longitudinal modes."], False),
    ("Full-Quantum-Dot Optoelectronic Memristors for Reservoir Computing",
     ["Here we develop an in-sensor reservoir computing system using quantum dot heterojunctions."], False),
    ("Reservoir computing using quantum dots",
     ["Here we implement a quantum reservoir computing system using entangled spin qubits in quantum dots."], True),
    ("Boson sampling powered image recognition",
     ["Here we show that boson sampling processors generate the dynamics necessary to power quantum reservoir computing."], True),
    ("Classical reservoir computing for quantum state tomography",
     ["We use a classical reservoir computing network to reconstruct quantum states from measurement data."], False),
    ("Control over quantum state tomography with reservoir computing networks",
     ["In our approach, the quantum reservoir is modeled with the Lindbladian equation.",
      "The control uses the coherent coupling between the input quantum state and the reservoir."], True),
    ("Experimental quantum tomography via quantum reservoir processing",
     ["Here we demonstrate a QRP approach on a bosonic circuit quantum electrodynamics platform."], True),
    ("Entanglement stabilization in a superconducting qutrit-qubit system",
     ["Quantum reservoir engineering is a framework for autonomous quantum state preparation.",
      "In this work we demonstrate Bell state stabilization using a dissipative bath."], False),
    ("Classical electronics for learning",
     ["Applications include quantum reservoir computing, optical sensing and image classification.",
      "Here we develop a classical electronic processor."], False),
    ("A new reservoir model",
     ["Applications include quantum reservoir computing and optical sensing.",
      "Here we demonstrate quantum reservoir computing using superconducting qubits."], True),
    ("A learning system",
     ["Quantum reservoir computing (QRC) provides a framework for temporal tasks.",
      "Here we investigate the learning capacity of QRC under decoherence."], True),
    ("Quantum Next Generation Reservoir Computing: An Efficient Quantum Algorithm", [], True),
    ("Quantum Noise-Induced Reservoir Computing", [], True),
    ("Hybrid quantum-classical reservoir computing",
     ["Two hybrid quantum-classical reservoir computing models are presented.",
      "Their performance is compared with a classical reservoir computing model."], True),
    ("Quantum reservoir complexity by Krylov evolution",
     ["Quantum reservoir computing algorithms use quantum dynamics.",
      "Our results show that the Krylov approach correlates with quantum reservoir performance."], True),
    ("Quantum reservoir engineering for state stabilization",
     ["Here we engineer a quantum reservoir using superconducting qubits for Bell state stabilization."], False),
    ("Reservoir computing using a Chua circuit for post-quantum cryptography",
     ["We implement a classical Chua circuit for post-quantum cryptography."], False),
    ("Quantum-Classical Hybrid Information Processing via a Single Quantum System",
     ["We propose a quantum reservoir processor to harness quantum dynamics in computational tasks."], True),
    ("Hilbert space as a computational resource in reservoir computing",
     ["With a reservoir comprised of a single quantum system, we demonstrate performance improvement and advantage over the classical reservoir."], True),
    ("Characterizing the memory capacity of transmon qubit reservoirs",
     ["Quantum Reservoir Computing (QRC) exploits the dynamics of quantum ensemble systems for machine learning.",
      "In this study, we focus on the task of characterizing the memory capacity of quantum reservoirs built using transmon devices provided by IBM."], True),
    ("Characterizing classical reservoir models of transmon qubit dynamics",
     ["We focus on characterizing a classical reservoir computing model to predict transmon qubit measurement data."], False),
])
def test_quantum_profile_distinguishes_quantum_execution_from_material_name_and_background(title, parts, expected):
    profile = directions.direction_profile("quantum reservoir computing")
    assert directions.profile_scope_check(profile, title, parts) is expected


@pytest.mark.parametrize("title,parts,expected", [
    ("Constrained Coding with Error Control for DNA-Based Data Storage", [], True),
    ("An Upper Bound on the Capacity of the DNA Storage Channel", [], True),
    ("Secure Cloud Data Storage Using DNA and Chaos Cryptography",
     ["We propose a DNA cryptographic algorithm using an encoding table and XOR to encrypt cloud data."], False),
    ("Enhanced DNA Cryptosystem for Secure Cloud Data Storage",
     ["In this paper we propose a DNA cryptosystem based on the chemical properties of DNA for cloud data storage."], False),
    ("DNA Cryptography for Secure Data Storage",
     ["We encode electronic data with an ACGT substitution table and store the encrypted file on a server."], False),
    ("Genomic Encryption of Digital Data Stored in Synthetic DNA",
     ["Here we encrypt digital information stored in synthetic DNA and retrieve the data by sequencing."], True),
    ("DNA cryptography for secure cloud data storage",
     ["Here we demonstrate physical DNA data storage: encoded oligonucleotides are synthesized and sequenced."], True),
    ("DNA neural activation functions",
     ["DNA computing has applications in data storage, diagnostics and neural circuits.",
      "This paper proposes a new activation function implemented by DNA hybridization."], False),
    ("Functions and applications of enzymes in nucleic acid nanotechnology",
     ["DNA nanotechnology includes data storage and biosensing.",
      "Here we develop enzymes for biological diagnostics."], False),
    ("DNA technology for computing",
     ["DNA technology includes data storage and biosensing.",
      "Here we propose a DNA data storage channel and derive its coding capacity."], True),
    ("Nucleic Acid Databases and Molecular-Scale Computing",
     ["Several robust DNA storage architectures featuring random access have been constructed.",
      "We survey these recent achievements and discuss engineering practical DNA storage systems."], True),
    ("Non-binary Codes for Correcting a Burst of Deletions",
     ["Deletion errors are common in DNA data storage.",
      "In this paper we construct non-binary codes to correct deletion bursts in DNA synthesis."], True),
    ("A channel coding theorem",
     ["DNA data storage is one motivation for deletion channels.",
      "We construct codes for this channel with improved redundancy."], True),
    ("DNA diagnostic circuits",
     ["DNA data storage requires error-correcting codes.",
      "Here we construct a DNA diagnostic circuit to detect RNA viruses."], False),
    ("New molecular memory",
     ["Here we construct synthetic DNA strands with addressable sequences.",
      "This system stores digital information and enables random access."], True),
    ("DNA punch cards for storing data on native DNA sequences", [], True),
    ("Layer-by-Layer DNA Encapsulated in Magnetic Nanoparticles",
     ["In this paper the practical density of long-term DNA storage is increased."], True),
    ("Image Encoding Using Multi-Level DNA Barcodes with Nanopore Readout",
     ["Herein, a DNA nanostructure-based storage method to save a grayscale image is proposed."], True),
    ("Sequential DNA Coding for Programmable Information Encryption",
     ["DNA molecules are a material for information storage and data encryption.",
      "This study introduces a programmable encryption strategy based on long-chain DNA synthesis.",
      "The proposed system enables the recording of encoded information."], True),
    ("On Coding Over Sliced Information",
     ["Channel models have applications in DNA storage, among others.",
      "In this paper we analyze the redundancy of binary codes for this channel."], True),
    ("DNA-based Authentication for Securing Cloud Data Storage",
     ["This study introduces a cryptographic key system exploiting synthetic DNA information capacity.",
      "The system authenticates users and stores their encrypted files on the cloud."], False),
    ("Secure DNA Data Storage with Similarity Search in Cloud Environments",
     ["We propose efficient similarity search over encrypted genomic DNA data stored on a cloud server."], False),
    ("DNA nanotechnology in cancer drug delivery",
     ["We investigate DNA computation for logic gates, DNA circuits, data storage, and machine learning."], False),
    ("Sequential DNA Coding for Programmable Information Encryption",
     ["This study introduces encryption based on long-chain DNA synthesis.",
      "The proposed hairpin-mediated primer exchange reaction system enables the recording of encoded information."], True),
    ("Sequential DNA Coding for Programmable Information Encryption",
     ["This study introduces a programmable encryption strategy based on long-chain DNA synthesis and sequential encoding.",
      "The proposed hairpin-mediated primer exchange reaction (HAMER) system enables the generation of DNA keys and the recording of encoded information."], True),
    ("DNA Methylation (DM) data format and DMtools for efficient DNA methylation data storage and analysis",
     ["In this study, we present a compressed binary format for storing DNA methylation data after mapping."], False),
    ("DNA methylation data storage",
     ["Here we store digital data in synthetic DNA and read the encoded files by sequencing."], True),
    ("Optimizing mycobacteria molecular diagnostics and DNA storage",
     ["We consider the best Mycobacterium tuberculosis DNA storage method to optimize molecular testing."], False),
])
def test_dna_profile_preserves_physical_storage_and_coding_but_rejects_background_or_symbolic_crypto(title, parts, expected):
    profile = directions.direction_profile("dna data storage")
    assert directions.profile_scope_check(profile, title, parts) is expected
