from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from orchestration.models.core import Artifact
from orchestration.models.factory import create_model
from orchestration.settings import settings


@dataclass
class DatabaseDeps:
    agent_name: str = "database"


database_agent = Agent(
    create_model(settings.resolve_model(settings.database_model)),
    deps_type=DatabaseDeps,
    system_prompt="""You are a Senior Database Engineer.
Your objective is to generate optimal SQL migrations, schema DDL, Redis/Cache configurations, and data seeding scripts.
Ensure databases are normalized where appropriate, properly indexed, and follow best practices for the chosen RDBMS (e.g., PostgreSQL).
Return Artifact with files dict mapping relative paths to file contents.


CRITICAL WORKFLOW:
1. **History Review:** Use `query_history` to recall previous architectural and contextual decisions before generating logic.
2. **Log Progress:** Call `log_activity` after significant actions.
3. **Proactive Docs:** Keep `docs/` updated via `update_documentation` when implementing major features.
""",
    output_type=Artifact,
    defer_model_check=True,
    model_settings={"max_tokens": 8192},
    retries=5,
)


@database_agent.tool
def generate_migration_template(ctx: RunContext[str], table_name: str, schema_description: str) -> str:
    """Provides a bulletproof PostgreSQL migration template including auto-updating timestamps."""
    return f"""-- Migration for {table_name}
-- Description: {schema_description}

-- UP
-- Ensure the updated_at trigger function exists (usually done in an initial setup migration, but safe to repeat)
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TABLE IF NOT EXISTS {table_name} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    -- LLM: Insert specific domain columns here
);

CREATE TRIGGER update_{table_name}_modtime
    BEFORE UPDATE ON {table_name}
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

CREATE INDEX IF NOT EXISTS idx_{table_name}_created_at ON {table_name}(created_at);

-- DOWN
DROP TRIGGER IF EXISTS update_{table_name}_modtime ON {table_name};
DROP TABLE IF EXISTS {table_name};
"""


@database_agent.tool
def generate_seed_script(ctx: RunContext[str], table_name: str) -> str:
    """Returns a template for a PostgreSQL data seeding script."""
    return f"""-- Seed data for {table_name}
INSERT INTO {table_name} (id, created_at, updated_at /* add columns */)
VALUES
    (gen_random_uuid(), NOW(), NOW()),
    (gen_random_uuid(), NOW(), NOW())
ON CONFLICT (id) DO NOTHING;
"""


@database_agent.tool
def scaffold_redis_config(ctx: RunContext[str]) -> str:
    """Returns a baseline Redis configuration for caching."""
    return """# Redis baseline configuration
maxmemory 256mb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
"""
