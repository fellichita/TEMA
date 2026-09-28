"""Shared, frozen TF-IDF/NMF fit for discovery and temporal validation."""

import math
import warnings

from app.ml.corpus import checkpoint


GENERIC = {"using", "based", "study", "paper", "results", "proposed", "approach", "method", "methods", "work",
           "new", "show", "demonstrate", "demonstrated", "performance", "high", "low", "different", "novel",
           "recent", "abstract", "applications", "application", "potential", "review", "introduction", "et", "al"}


def texts_for_studies(studies):
    return [s["title"] + ". " + s["title"] + ". " + s["abstract"][:6000] for s in studies]


def fit_topics(studies, cancel=None):
    """Fit only supplied studies; callers must perform temporal splitting first."""
    from sklearn.decomposition import NMF
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from threadpoolctl import threadpool_limits

    checkpoint(cancel)
    vectorizer = TfidfVectorizer(stop_words=sorted(set(ENGLISH_STOP_WORDS) | GENERIC), ngram_range=(1, 3),
                                 min_df=2, max_df=0.9, max_features=18_000, sublinear_tf=True,
                                 token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b")
    matrix = vectorizer.fit_transform(texts_for_studies(studies))
    checkpoint(cancel)
    n_topics = min(32, max(2, int(math.sqrt(len(studies)))), matrix.shape[0] - 1, matrix.shape[1])
    if n_topics < 1:
        raise ValueError("Недостаточно документов для тематической модели.")
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=250, tol=0.001)
        memberships = model.fit_transform(matrix)
    checkpoint(cancel)
    return {"matrix": matrix, "vectorizer": vectorizer, "model": model, "memberships": memberships,
            "n_topics": n_topics,
            "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught)}
