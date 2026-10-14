from unittest.mock import patch

from pydantic_agents.core.healthcheck import validate_api_keys, validate_directories


class TestValidateAPIKeys:
    def test_no_keys_returns_error(self):
        with patch("pydantic_agents.core.healthcheck.settings") as mock:
            mock.openai_api_key = None
            mock.anthropic_api_key = None
            mock.gemini_api_key = None
            errors = validate_api_keys()
            assert any("No API keys configured" in e for e in errors)

    def test_placeholder_detected(self):
        with patch("pydantic_agents.core.healthcheck.settings") as mock:
            mock.openai_api_key = "sk-..."
            mock.anthropic_api_key = None
            mock.gemini_api_key = None
            errors = validate_api_keys()
            assert any("placeholder" in e.lower() for e in errors)

    def test_valid_key_passes(self):
        with patch("pydantic_agents.core.healthcheck.settings") as mock:
            mock.openai_api_key = "sk-real-key"
            mock.anthropic_api_key = None
            mock.gemini_api_key = None
            errors = validate_api_keys()
            assert len(errors) == 0


class TestValidateDirectories:
    def test_writable_directory_passes(self, tmp_path):
        with patch("pydantic_agents.core.healthcheck.settings") as mock:
            mock.output_dir = tmp_path / "out"
            mock.checkpoint_dir = tmp_path / "ckpt"
            mock.vector_db_dir = tmp_path / "vdb"
            errors = validate_directories()
            assert len(errors) == 0

    def test_missing_directory_parent_reported(self, tmp_path):
        deep_dir = tmp_path / "does-not-exist" / "yet"
        with patch("pydantic_agents.core.healthcheck.settings") as mock:
            mock.output_dir = deep_dir
            mock.checkpoint_dir = tmp_path / "ckpt"
            mock.vector_db_dir = tmp_path / "vdb"
            errors = validate_directories()
            assert any("does not exist or is not writable" in e for e in errors)

    def test_missing_dir_with_writable_parent_passes(self, tmp_path):
        parent = tmp_path / "exists"
        parent.mkdir()
        child = parent / "newdir"
        with patch("pydantic_agents.core.healthcheck.settings") as mock:
            mock.output_dir = child
            mock.checkpoint_dir = tmp_path / "ckpt"
            mock.vector_db_dir = tmp_path / "vdb"
            errors = validate_directories()
            assert len(errors) == 0
