"""The generated Dockerfile, CI, Makefile, compose file and Helm chart agree with each other."""

from unittest.mock import MagicMock

import pytest
import yaml

from pydantic_agents.agents.backend import generate_dockerfile
from pydantic_agents.agents.database import generate_migration_template
from pydantic_agents.agents.devops import generate_docker_compose, generate_github_action, generate_makefile
from pydantic_agents.agents.sre import generate_helm_chart
from pydantic_agents.core.settings import settings

CTX = MagicMock()
PKG = settings.go_entrypoint.rsplit("/", 1)[0]


@pytest.mark.parametrize(
    "render",
    [
        lambda: generate_dockerfile(CTX, "orders", 8080),
        lambda: generate_makefile(CTX, "orders"),
        lambda: generate_github_action(CTX, "orders"),
    ],
)
def test_build_templates_use_the_configured_entrypoint(render):
    text = render()
    assert PKG in text
    assert "cmd/orders" not in text


def test_ci_pins_tool_versions():
    text = generate_github_action(CTX, "orders")
    assert "@latest" not in text and "version: latest" not in text
    assert settings.golangci_lint_version in text and settings.govulncheck_version in text


def test_compose_uses_the_current_format():
    text = generate_docker_compose(CTX, "orders")
    assert "version:" not in text
    assert "docker-compose" not in generate_makefile(CTX, "orders")
    assert "orders" in yaml.safe_load(text)["services"]


def test_dockerfile_is_a_real_multistage_build():
    text = generate_dockerfile(CTX, "orders", 8080)
    assert text.count("FROM ") == 2 and "USER app" in text and "HEALTHCHECK" in text
    assert "# Build steps" not in text


def test_migration_trigger_is_idempotent():
    sql = generate_migration_template(CTX, "orders", "d")
    assert sql.index("DROP TRIGGER IF EXISTS") < sql.index("CREATE TRIGGER")


def test_helm_chart_has_real_content():
    files = generate_helm_chart(CTX, "orders", "repo/orders", 8080)
    assert {f.rsplit("/", 1)[-1] for f in files} >= {"Chart.yaml", "values.yaml", "deployment.yaml", "hpa.yaml"}
    assert all(len(body.strip().splitlines()) > 1 and not body.lstrip().startswith("#") for body in files.values())
    assert yaml.safe_load(files["helm/orders/Chart.yaml"])["name"] == "orders"
    assert yaml.safe_load(files["helm/orders/values.yaml"])["service"]["port"] == 8080
