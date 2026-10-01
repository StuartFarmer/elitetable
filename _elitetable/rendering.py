"""Render packaged templates with the archived Slick 0.3.0 request format."""

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import TypeAdapter
from slick.prompts import parse


def render(template, output_type, **variables):
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).with_name("prompts")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    schema = TypeAdapter(output_type).json_schema()
    format_block = (
        "# Output Format\n\n"
        "Respond with ONLY a JSON value matching this schema — no preamble, no "
        "commentary, nothing outside the JSON.\n\n"
        f"{json.dumps(schema, indent=2)}\n"
    )
    text = environment.get_template(template).render(
        **variables, schema=schema, output_format=format_block,
    ).strip()
    # Slick 0.3.0 appends this block even when the template already embeds schema.
    if "# Output Format" not in text:
        text = f"{text}\n\n---\n\n{format_block}"
    return text


async def generate(template, output_type, provider, **variables):
    text, _ = await provider.acall(render(template, output_type, **variables))
    return parse(text, output_type)
