"""Configuration-relative paths, independent of numerical runtime dependencies."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def project_path(settings, value):
    """Resolve a configured directory or file against its configuration folder."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(settings.get('_project_dir', ROOT)) / path


def input_path(settings, key):
    """Resolve an input override or a path beneath the configured research root."""
    path = Path(settings['paths'][key]).expanduser()
    return path if path.is_absolute() else project_path(settings, settings['research_dir']) / path
