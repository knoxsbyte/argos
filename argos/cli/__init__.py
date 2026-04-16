"""ARGOS CLI package — exports ArgosApp (Textual) and app (Typer)."""

from argos.cli.app import ArgosApp


def _get_typer_app():
    from argos.main import app
    return app


__all__ = ["ArgosApp"]
