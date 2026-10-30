import logging
import os
from pathlib import Path

from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)


PLACEHOLDER = [
    "sk-...",
    "sk-ant-...",
    "API_KEY",
    "api-key",
]


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return False
    stripped = value.strip()
    return any(p in stripped for p in PLACEHOLDER)


def validate_api_keys() -> list[str]:
    """Check that at least one API key is configured and not a placeholder."""
    errors: list[str] = []

    if settings.model_provider == "litellm":
        litellm_key = next(
            (
                v
                for v in (
                    os.environ.get("LITELLM_VIRTUAL_KEY"),
                    os.environ.get("LITELLM_MASTER_KEY"),
                    getattr(settings, "litellm_virtual_key", None),
                    getattr(settings, "litellm_master_key", None),
                )
                if isinstance(v, str) and v
            ),
            None,
        )
        if not litellm_key or _is_placeholder(litellm_key):
            errors.append("Model provider is 'litellm' but LITELLM_VIRTUAL_KEY or LITELLM_MASTER_KEY is not set.")
        return errors

    keys = {
        "OPENAI_API_KEY": settings.openai_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "GEMINI_API_KEY": settings.gemini_api_key,
        "DEEPSEEK_API_KEY": settings.deepseek_api_key,
    }
    configured = []
    for name, value in keys.items():
        if isinstance(value, str) and value and not _is_placeholder(value):
            configured.append(name)
        elif isinstance(value, str) and _is_placeholder(value):
            errors.append(f"{name} appears to be a placeholder value. Set a real key in .env")
    if not configured:
        errors.append(
            "No API keys configured. Set at least one of: "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY"
        )
    return errors


def validate_directories() -> list[str]:
    """Ensure output/checkpoint directories are writable (read-only check, no FS mutation)."""
    errors: list[str] = []
    for name, path in [
        ("OUTPUT_DIR", settings.output_dir),
        ("CHECKPOINT_DIR", settings.checkpoint_dir),
    ]:
        try:
            resolved = Path(path).resolve()
            if resolved.exists():
                if not os.access(str(resolved), os.W_OK):
                    errors.append(f"{name} ({resolved}) exists but is not writable")
            else:
                parent = resolved.parent
                if not parent.exists() or not os.access(str(parent), os.W_OK):
                    errors.append(f"{name} ({resolved}) parent directory does not exist or is not writable")
        except OSError as e:  # PermissionError is a subclass
            errors.append(f"{name}={path}: {e}")
    return errors


def validate_database() -> list[str]:
    """Check pgvector database connectivity (async safe)."""
    errors: list[str] = []
    dsn = settings.database_url or os.environ.get("DATABASE_URL", "")
    if not dsn:
        errors.append("DATABASE_URL not set — pgvector vector store unavailable")
        return errors

    async def _check():
        try:
            import asyncpg

            conn = await asyncpg.connect(dsn, timeout=5)
            row = await conn.fetchrow("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            if row is None:
                errors.append("pgvector extension not installed in database")
            else:
                logger.info("pgvector extension version: %s", row["extversion"])
            await conn.close()
        except Exception as e:
            errors.append(f"Database connection failed: {e}")
        return errors

    import asyncio

    try:
        asyncio.get_running_loop()
        # already inside an event loop — can't call asyncio.run()
        logger.warning("Skipping database validation: already inside async context")
        return errors
    except RuntimeError:
        return asyncio.run(_check())


def run_startup_checks() -> list[str]:
    """Run all startup checks and return a list of error messages."""
    errors: list[str] = []
    errors.extend(validate_api_keys())
    errors.extend(validate_directories())
    errors.extend(validate_database())
    if errors:
        logger.debug("startup checks found %d issue(s): %s", len(errors), errors)
    return errors


def assert_startup_checks():
    """Run startup checks and log warnings for any issues found."""
    errors = run_startup_checks()
    for err in errors:
        logger.warning("Startup check failed: %s", err)
    if errors:
        logger.warning("%d startup issue(s) found. The pipeline may fail at runtime.", len(errors))
    return errors
