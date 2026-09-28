"""Запуск из корня проекта: python -m app.main."""

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    from app.runtime.session import credentials

    credentials()
    from app.ui.window import run_app

    run_app()
