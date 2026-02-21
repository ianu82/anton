from __future__ import annotations

from pathlib import Path

import pytest

from anton.skill.base import SkillInfo
from anton.skill.loader import load_skill_module

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


class TestLoadSkillModule:
    def test_load_read_file_skill(self):
        path = SKILLS_DIR / "read_file" / "skill.py"
        skills = load_skill_module(path)
        assert len(skills) >= 1
        info = skills[0]
        assert isinstance(info, SkillInfo)
        assert info.name == "read_file"
        assert info.source_path == path

    def test_load_write_file_skill(self):
        path = SKILLS_DIR / "write_file" / "skill.py"
        skills = load_skill_module(path)
        assert len(skills) >= 1
        assert skills[0].name == "write_file"

    def test_load_nonexistent_path(self):
        path = Path("/does/not/exist/skill.py")
        # Broken/missing skills are skipped gracefully
        result = load_skill_module(path)
        assert result == []

    def test_loaded_skill_has_parameters(self):
        path = SKILLS_DIR / "run_command" / "skill.py"
        skills = load_skill_module(path)
        info = skills[0]
        assert "command" in info.parameters["properties"]
        assert "command" in info.parameters["required"]

    def test_loaded_skill_is_callable(self):
        path = SKILLS_DIR / "list_files" / "skill.py"
        skills = load_skill_module(path)
        info = skills[0]
        assert callable(info.execute)


class TestSkillErrorFile:
    def test_error_file_written_on_broken_skill(self, tmp_path: Path):
        skill_dir = tmp_path / "bad_skill"
        skill_dir.mkdir()
        skill_path = skill_dir / "skill.py"
        skill_path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

        result = load_skill_module(skill_path)

        assert result == []
        error_path = skill_dir / "skill.error"
        assert error_path.exists()
        contents = error_path.read_text(encoding="utf-8")
        assert "RuntimeError" in contents
        assert "boom" in contents

    def test_error_file_cleaned_on_success(self, tmp_path: Path):
        skill_dir = tmp_path / "good_skill"
        skill_dir.mkdir()
        skill_path = skill_dir / "skill.py"
        # Write a valid module (no skills, but loads fine)
        skill_path.write_text("x = 1\n", encoding="utf-8")
        # Pre-create a stale error file
        error_path = skill_dir / "skill.error"
        error_path.write_text("old error\n", encoding="utf-8")

        load_skill_module(skill_path)

        assert not error_path.exists()

    def test_error_file_not_created_on_success(self, tmp_path: Path):
        skill_dir = tmp_path / "clean_skill"
        skill_dir.mkdir()
        skill_path = skill_dir / "skill.py"
        skill_path.write_text("x = 1\n", encoding="utf-8")

        load_skill_module(skill_path)

        error_path = skill_dir / "skill.error"
        assert not error_path.exists()
