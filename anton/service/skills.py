from __future__ import annotations

from string import Formatter


def extract_template_fields(prompt_template: str) -> list[str]:
    fields: set[str] = set()
    for _literal, field_name, _format_spec, _conversion in Formatter().parse(prompt_template):
        if field_name is None:
            continue
        field = field_name.strip()
        if not field:
            continue
        if "." in field or "[" in field or "]" in field:
            raise ValueError(f"Unsupported placeholder '{field}'. Use simple names like '{{metric}}'.")
        fields.add(field)
    return sorted(fields)


def render_prompt_template(prompt_template: str, *, required_params: list[str], params: dict[str, str] | None) -> str:
    provided = dict(params or {})
    missing = [field for field in required_params if field not in provided]
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"Missing required skill params: {missing_csv}")
    try:
        rendered = prompt_template.format(**provided)
    except KeyError as exc:
        raise ValueError(f"Missing required skill param: {exc.args[0]}") from exc
    if not rendered.strip():
        raise ValueError("Rendered skill prompt is empty.")
    return rendered
