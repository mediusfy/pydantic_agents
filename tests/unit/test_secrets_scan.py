"""Unit tests for secret scanning and snippet redaction."""

from unittest.mock import MagicMock

from pydantic_agents.core.tools.secrets import scan_for_secrets


def test_scan_for_secrets_redacts_matches_and_catches_expanded_patterns():
    sample = "\n".join(
        [
            "OPENAI_API_KEY = sk-proj-abcdefghijklmnopqrstuvwxyz123456",
            "api_key = my_super_secret_token_value_12345",
            "DATABASE_URL=postgresql://admin:s3cr3tP@ss@db.prod.internal:5432/app-db",
            "REDIS_URL=rediss://default:r3d1sPass@cache-01.internal:6379",
        ]
    )
    findings = scan_for_secrets(MagicMock(), sample)
    types = {f["type"] for f in findings}
    assert {"llm_api_key", "api_key_generic", "pg_connection_string", "redis_connection_string"} <= types
    for finding in findings:
        assert "s3cr3tP@ss" not in finding["snippet"]
        assert "r3d1sPass" not in finding["snippet"]
        assert "abcdefghijklmnopqrstuvwxyz123456" not in finding["snippet"]
        assert "my_super_secret_token_value_12345" not in finding["snippet"]
        assert "[REDACTED]" in finding["snippet"]
