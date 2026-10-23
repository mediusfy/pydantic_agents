import re
from typing import Any

from pydantic_ai import RunContext

from pydantic_agents.core.tools.registry import register_tool

PATTERNS = [
    # Cloud & Infrastructure
    ("aws_access_key", r"AKIA[0-9A-Z]{16}", "high"),
    ("aws_secret_key", r"[aA][wW][sS](.{0,20})?['\"]?[0-9a-zA-Z\/+]{40}['\"]?", "critical"),
    ("aws_session_token", r"ASIA[0-9A-Z]{16}", "high"),
    ("gcp_api_key", r"AIza[0-9A-Za-z\-_]{35}", "high"),
    ("gcp_service_account_json", r"(?i)[\"']type[\"']\s*:\s*[\"']service_account[\"']", "high"),
    # Repositories & Private Keys
    ("github_token", r"(?:ghp|gho|ghu|ghs|ghr)_[0-9a-zA-Z]{36}", "critical"),
    ("github_old_token", r"github_pat_[0-9a-zA-Z]{22,}", "critical"),
    ("private_key", r"-----BEGIN\s?(?:RSA|DSA|EC|OPENSSH|PGP)?\s?PRIVATE KEY-----", "critical"),
    # General API & Tokens ---
    (
        "api_key_generic",
        r"(?i)(api[_-]?key|apikey|secret[_-]?key|app[_-]?secret)[:=]\s*['\"][0-9a-zA-Z_\-]{16,}['\"]",
        "high",
    ),
    ("bearer_token", r"(?i)bearer\s+[0-9a-zA-Z\-_\.]{20,}", "high"),
    ("password_in_code", r"(?i)(password|passwd|pwd)[:=]\s*['\"][^'\"]{6,}['\"]", "high"),
    ("jwt_token", r"eyJ[0-9a-zA-Z\-_]+\.eyJ[0-9a-zA-Z\-_]+\.[0-9a-zA-Z\-_]+", "medium"),
    ("npm_token", r"npm_[a-z0-9A-Z]{36}", "high"),
    # Communications & SaaS
    ("slack_token", r"xox[baprs]-[0-9a-zA-Z\-]{10,}", "high"),
    (
        "slack_webhook",
        r"https://hooks\.slack\.com/services/T[a-zA-Z0-9_]{8,}/B[a-zA-Z0-9_]{8,}/[a-zA-Z0-9_]{24}",
        "critical",
    ),
    ("discord_webhook", r"https://discord\.com/api/webhooks/[0-9]{17,19}/[a-zA-Z0-9_-]{68}", "high"),
    ("discord_bot_token", r"[M-Z][a-zA-Z0-9_-]{23,25}\.[a-zA-Z0-9_-]{6}\.[a-zA-Z0-9_-]{27}", "high"),
    ("twilio_account_sid", r"AC[a-zA-Z0-9]{32}", "critical"),
    ("sendgrid_api_key", r"SG\.[0-9a-zA-Z_-]{22}\.[0-9a-zA-Z_-]{43}", "critical"),
    ("mailgun_api_key", r"key-[0-9a-zA-Z]{32}", "high"),
    # Payment Gateways
    ("stripe_key", r"(?:sk|rk)_(?:test|live)_[0-9a-zA-Z]{24}", "critical"),
    ("square_access_token", r"sqOatp-[0-9A-Za-z\-_]{22}", "high"),
    # Databases
    ("pg_connection_string", r"postgres(ql)?://[^:]+:[^@]+@\w+:\d+/\w+", "critical"),
    ("redis_connection_string", r"redis://:[^@]+@\w+:\d+", "critical"),
    ("mongodb_connection_string", r"mongodb(\+srv)?://[^:]+:[^@]+@[\w\.-]+/\w+", "critical"),
]


async def scan_for_secrets(ctx: RunContext[Any], content: str) -> list[dict]:
    findings = []
    for name, pattern, severity in PATTERNS:
        for i, line in enumerate(content.splitlines(), 1):
            if re.search(pattern, line):
                findings.append(
                    {
                        "type": name,
                        "line": i,
                        "snippet": line.strip()[:120],
                        "severity": severity,
                    }
                )
    return findings


register_tool(
    "scan_for_secrets",
    scan_for_secrets,
    "Scan content for hardcoded secrets, API keys, and credentials",
    ["security", "governance"],
)
