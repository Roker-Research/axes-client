"""Jinja prompt loading, modeled on Chat Plot's ``load_prompts``.

Templates live next to the agent's source file as ``<key>.md`` and are
rendered with ``StrictUndefined`` so a missing variable fails loudly. Loading
is tolerant of *missing* templates: a procedural agent that overrides
``plan_step`` need not ship a ``prompt.md``.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    Template,
    TemplateNotFound,
)


def load_prompts(source_file: str, keys: list[str]) -> dict[str, Template]:
    directory = Path(source_file).parent
    env = Environment(
        loader=FileSystemLoader(directory),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    templates: dict[str, Template] = {}
    for key in keys:
        try:
            templates[key] = env.get_template(f"{key}.md")
        except TemplateNotFound:
            continue
    return templates
