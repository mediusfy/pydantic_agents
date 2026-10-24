import re
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_agents.core.models.core import (
    SECURITY_CRITICAL_MAX_SCORE,
    SECURITY_HIGH_MAX_SCORE,
    SECURITY_PASS_SCORE,
    SecurityReport,
)
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings


@dataclass
class SecurityDeps:
    agent_name: str = "security"


security_prompt = """You are a Principal Security Engineer.

Your objective is to comprehensively audit generated or existing code for security vulnerabilities.

## Audit Scope

Check for:
- **OWASP Top 10** vulnerabilities
- **Hardcoded secrets**: API keys, tokens, passwords, connection strings in code
- **SQL injection**: unsanitized inputs in database queries
- **XSS**: unsanitized user input rendered in output
- **SSRF**: user-controlled URLs in server-side requests
- **Insecure deserialization**: untrusted data deserialized without validation
- **Weak cryptography**: MD5, SHA1, hardcoded keys, short key lengths
- **Broken authn/authz**: missing authentication, privilege escalation paths
- **Insecure defaults**: debug mode enabled, permissive CORS, exposed admin endpoints

## Error Handling & Logging Security

- Error messages must not leak sensitive information (stack traces, DB queries, internal paths)
- Logging must not include secrets, PII, or tokens
- Use typed sentinel errors — never leak driver/SDK errors to callers

## Dependency Injection Security

- No global state that could leak across requests
- No `init()` functions with side effects that could hide security issues

## Output

Return a SecurityReport with:
- `score`: 0-100 (100 = clean), using these bands:
  - any CRITICAL finding: __CRITICAL_MAX__ or lower
  - any HIGH finding (and no critical): __HIGH_MAX__ or lower
  - only medium/low findings, or none: __PASS__ or higher
- `passed`: do not worry about getting this right; it is derived from the score (true iff score is __PASS__ or higher)
- `feedback`: specific findings with file:line, vulnerability type, severity, and remediation
"""


def _render_prompt(template: str, values: dict[str, int]) -> str:
    """Substitute __NAME__ placeholders, failing at import time if one is missing from the template or left over."""
    for name, value in values.items():
        if f"__{name}__" not in template:
            raise ValueError(f"security prompt has no __{name}__ placeholder")
        template = template.replace(f"__{name}__", str(value))
    leftover = re.findall(r"__[A-Z_]+__", template)
    if leftover:
        raise ValueError(f"security prompt has unfilled placeholders: {leftover}")
    return template


security_prompt = _render_prompt(
    security_prompt,
    {
        "CRITICAL_MAX": SECURITY_CRITICAL_MAX_SCORE,
        "HIGH_MAX": SECURITY_HIGH_MAX_SCORE,
        "PASS": SECURITY_PASS_SCORE,
    },
)

security_agent = Agent(
    create_model(settings.resolve_model(settings.security_model)),
    deps_type=SecurityDeps,
    system_prompt=security_prompt,
    output_type=SecurityReport,
    defer_model_check=True,
    retries=settings.agent_retries,
)
