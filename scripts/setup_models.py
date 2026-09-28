"""Prepare pinned local models: python -m scripts.setup_models [--web-analysis] [--profile DIR].

The default setup stages the three ONNX models used across the application under
`storage/` and copies them to the paths the application reads. `--web-analysis`
selects just the scientific encoder, Russian translator, and local instruct model
needed by the web analysis. The instruct model is kept in the profile rather than
the project stage because its 1.8 GB weights are specific to that profile.
Existing verified files are reused; downloads are verified before publication.

Nothing here decides anything scientific. The three ONNX folders are reused
between profiles; the local instruct model stays in the profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, NamedTuple

from scripts.model_staging import install_from_staging, stage_from, staging_directory, verified_staged_copy


class Model(NamedTuple):
    key: str
    title: str
    spec: dict[str, Any]
    verify: Callable[..., Any]
    runtime: Path
    download: Callable[..., Any]
    staged: bool = True
    reuse_source: Path | None = None


def _models(profile: Path | None, *, web_analysis: bool = False) -> list[Model]:
    from app.identity import default_data_dir
    from app.ml import local_encoder
    from app.pilot import encoder as pilot_encoder
    from app.pilot import local_llm
    from app.pilot import translator
    from scripts.install_local_llm import install as install_local_llm
    from scripts.install_ml_model import install as install_ml
    from scripts.install_pilot_model import install as install_pilot
    from scripts.install_translation_model import install as install_translation

    data_dir = Path(profile) if profile is not None else default_data_dir()
    default_profile = default_data_dir()
    scientific = Model("multilingual-e5-small", "научный анализ", pilot_encoder.load_spec(),
                       pilot_encoder.verify_artifacts, pilot_encoder.model_directory(data_dir), install_pilot,
                       reuse_source=pilot_encoder.model_directory(default_profile) if web_analysis else None)
    translation = Model("opus-mt-ru-en", "перевод русского запроса", translator.load_spec(),
                        translator.verify_artifacts, translator.model_directory(data_dir), install_translation,
                        reuse_source=translator.model_directory(default_profile) if web_analysis else None)
    if web_analysis:
        instruct = Model(local_llm.MODEL_KEY, "локальный AI-анализ", local_llm.load_spec(),
                         local_llm.verify_artifacts, local_llm.model_directory(data_dir),
                         install_local_llm, staged=False,
                         reuse_source=local_llm.model_directory(default_profile))
        # Веб переводит ТОП на русский этой моделью; без неё карточки остаются в оригинале.
        # Её нет в проектном реестре (resources/models/registry.json), поэтому она
        # копируется прямо из основного профиля, минуя проектную копию.
        reading_key = translator.ENGLISH_RUSSIAN_KEY
        reading = Model(reading_key, "перевод ТОПа на русский", translator.load_spec(reading_key),
                        translator.verify_artifacts, translator.model_directory(data_dir, reading_key),
                        lambda directory, progress=None: install_translation(directory, model_key=reading_key),
                        staged=False,
                        reuse_source=translator.model_directory(default_profile, reading_key))
        return [scientific, translation, reading, instruct]
    return [
        scientific,
        Model("e5-small-v2", "смысловая проверка «Трендов»", local_encoder.load_spec(),
              local_encoder.verify_artifacts, Path(local_encoder.DEFAULT_MODEL_DIR), install_ml),
        translation,
    ]


def _ready(path: Path, model: Model) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    try:
        model.verify(path, model.spec)
    except Exception:
        return False
    return True


def _copy_verified(source: Path, target: Path, model: Model) -> bool:
    """Reuse pinned bytes in another profile without exposing a partial model."""
    if source == target or not _ready(source, model):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".verified-model-", dir=target.parent) as temporary:
        pending = Path(temporary) / "model"
        pending.mkdir()
        for item in model.spec["files"]:
            shutil.copyfile(source / item["name"], pending / item["name"])
        model.verify(pending, model.spec)
        if target.exists():
            if not target.is_dir() or target.is_symlink() or any(target.iterdir()):
                return False
            target.rmdir()
        pending.rename(target)
    return True


def prepare(model: Model, *, allow_download: bool,
            progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Make one model available for the running application, downloading only if unavoidable."""
    if not model.staged:
        if _ready(model.runtime, model):
            return {"key": model.key, "state": "ready", "actions": ["уже готово"],
                    "megabytes": round(sum(item["bytes"] for item in model.spec["files"]) / 1e6)}
        if model.reuse_source is not None and _copy_verified(model.reuse_source, model.runtime, model):
            return {"key": model.key, "state": "ready", "actions": ["скопировано из основного профиля"],
                    "megabytes": round(sum(item["bytes"] for item in model.spec["files"]) / 1e6)}
        if not allow_download:
            return {"key": model.key, "state": "missing", "actions": [],
                    "megabytes": round(sum(item["bytes"] for item in model.spec["files"]) / 1e6)}
        if progress is None:
            model.download(model.runtime)
        else:
            model.download(model.runtime, progress=progress)
        return {"key": model.key, "state": "ready" if _ready(model.runtime, model) else "runtime_blocked",
                "actions": ["скачано в профиль"],
                "megabytes": round(sum(item["bytes"] for item in model.spec["files"]) / 1e6)}
    staged = verified_staged_copy(model.key, model.spec, model.verify) is not None
    actions: list[str] = []
    if not staged and _ready(model.runtime, model):
        # The application already has it; keep a project copy so the next profile is free.
        stage_from(model.key, model.runtime, model.spec, model.verify)
        staged = True
        actions.append("сохранено в проект из профиля")
    if not staged and model.reuse_source is not None and _ready(model.reuse_source, model):
        stage_from(model.key, model.reuse_source, model.spec, model.verify)
        staged = verified_staged_copy(model.key, model.spec, model.verify) is not None
        if staged:
            actions.append("сохранено в проект из основного профиля")
        elif _copy_verified(model.reuse_source, model.runtime, model):
            # A damaged non-empty project stage is left untouched. The verified
            # default-profile copy can still make this web profile usable.
            return {"key": model.key, "state": "ready", "actions": ["скопировано из основного профиля"],
                    "megabytes": round(sum(item["bytes"] for item in model.spec["files"]) / 1e6)}
    if not staged:
        if not allow_download:
            return {"key": model.key, "state": "missing", "actions": actions,
                    "megabytes": round(sum(item["bytes"] for item in model.spec["files"]) / 1e6)}
        model.download(staging_directory(model.key))
        staged = verified_staged_copy(model.key, model.spec, model.verify) is not None
        actions.append("скачано в проект")
    if not _ready(model.runtime, model):
        if install_from_staging(model.key, model.runtime, model.spec, model.verify):
            actions.append("скопировано в профиль")
        else:
            return {"key": model.key, "state": "runtime_blocked", "actions": actions,
                    "megabytes": round(sum(item["bytes"] for item in model.spec["files"]) / 1e6)}
    return {"key": model.key, "state": "ready", "actions": actions or ["уже готово"],
            "megabytes": round(sum(item["bytes"] for item in model.spec["files"]) / 1e6)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, help="Каталог данных приложения; по умолчанию текущий профиль")
    parser.add_argument("--web-analysis", action="store_true",
                        help="Подготовить три модели для научного веб-анализа, включая локальный AI")
    parser.add_argument("--offline", action="store_true", help="Ничего не скачивать, только разложить локальное")
    parser.add_argument("--json", action="store_true", help="Машинный отчёт вместо таблицы")
    args = parser.parse_args(argv)

    report = []
    for model in _models(args.profile, web_analysis=args.web_analysis):
        if args.web_analysis and not args.json:
            print(f"Готовим модель: {model.title}...", flush=True)
        shown_decile = 0

        def show_progress(received: int, total: int) -> None:
            nonlocal shown_decile
            decile = min(10, max(0, received * 10 // total)) if total > 0 else 0
            if decile <= shown_decile:
                return
            shown_decile = decile
            print(f"    Загрузка локальной AI-модели: {received / 1e6:.0f} из {total / 1e6:.0f} МБ "
                  f"({decile * 10}%)", flush=True)

        try:
            row = prepare(model, allow_download=not args.offline,
                          progress=show_progress if args.web_analysis and not args.json and not model.staged else None)
        except Exception as error:  # noqa: BLE001 - отчёт важнее обрыва на первой модели
            row = {"key": model.key, "state": "failed", "error": type(error).__name__,
                   "message": str(error)[:200], "actions": []}
        row["title"] = model.title
        row["runtime"] = str(model.runtime)
        row["staging"] = str(staging_directory(model.key)) if model.staged else None
        report.append(row)

    if args.json:
        print(json.dumps({"models": report}, ensure_ascii=False, indent=1))
    else:
        for row in report:
            mark = {"ready": "готово", "missing": "не найдено", "failed": "ошибка",
                    "runtime_blocked": "каталог занят"}[row["state"]]
            print(f"{row['title']:<30}{mark:<14}{'; '.join(row['actions'])}")
            if row["state"] == "failed":
                print(f"    {row['error']}: {row['message']}")
            elif row["state"] != "ready":
                print(f"    профиль: {row['runtime']}")
        ready = sum(row["state"] == "ready" for row in report)
        total = sum(row["megabytes"] for row in report if "megabytes" in row)
        location = "в профиле и проекте" if args.web_analysis else "в проекте"
        print(f"\nГотово моделей: {ready} из {len(report)}; всего {total} МБ {location}.")
        if ready == len(report):
            print("Повторный запуск с тем же профилем скачиваний не требует.")
    return 0 if all(row["state"] == "ready" for row in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
