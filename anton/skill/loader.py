from __future__ import annotations

import importlib.util
import logging
import sys
import traceback
from pathlib import Path

from anton.skill.base import SkillInfo

logger = logging.getLogger(__name__)


def load_skill_module(path: Path) -> list[SkillInfo]:
    error_path = path.with_suffix(".error")
    module_name = f"anton_skill_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        logger.warning("Skipping broken skill %s: %s", path, exc)
        sys.modules.pop(module_name, None)
        try:
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass  # parent dir may not exist (e.g. nonexistent path)
        return []

    # Successful load — remove stale error file if present
    error_path.unlink(missing_ok=True)

    skills: list[SkillInfo] = []
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        info = getattr(obj, "_skill_info", None)
        if isinstance(info, SkillInfo):
            info.source_path = path
            skills.append(info)

    return skills
