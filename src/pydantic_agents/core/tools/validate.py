import json
import re
from typing import Any

from pydantic_ai import RunContext

from pydantic_agents.core.tools.registry import register_tool

try:
    import yaml
except ImportError:
    yaml = None


def validate_protobuf(ctx: RunContext[Any], content: str) -> tuple[bool, list[str]]:
    errors = []
    if not content.strip():
        errors.append("Empty protobuf file")
        return False, errors
    if "syntax" not in content:
        errors.append("Missing syntax declaration (e.g., syntax='proto3')")
    if "package " not in content:
        errors.append("Missing package declaration")
    msg_count = len(re.findall(r"\bmessage\s+\w+", content))
    if msg_count == 0:
        errors.append("No message definitions found")
    service_count = len(re.findall(r"\bservice\s+\w+", content))
    if service_count == 0:
        errors.append("No service definitions found")
    return len(errors) == 0, errors


def validate_yaml(ctx: RunContext[Any], content: str) -> tuple[bool, list[str]]:
    if yaml is None:
        return True, ["yaml library not available, skipping validation"]
    try:
        yaml.safe_load(content)
        return True, []
    except yaml.YAMLError as e:
        return False, [str(e)]


def validate_json(ctx: RunContext[Any], content: str) -> tuple[bool, list[str]]:
    try:
        json.loads(content)
        return True, []
    except json.JSONDecodeError as e:
        return False, [f"line {e.lineno}: {e.msg}"]


def parse_go_ast(ctx: RunContext[Any], content: str) -> tuple[bool, list[str]]:
    errors = []
    if not content.strip():
        errors.append("Empty Go file")
        return False, errors
    if "package " not in content:
        errors.append("Missing package declaration")
    imports = re.findall(r"import\s+\(", content)
    unclosed = re.findall(r"import\s+\([^)]*$", content, re.MULTILINE)
    if imports and unclosed:
        errors.append("Unclosed import block")
    func_count = len(re.findall(r"\bfunc\s+\w+", content))
    if func_count == 0 and content.count("package ") > 0:
        errors.append("No function definitions found (possibly empty scaffold)")
    brace_count = content.count("{") - content.count("}")
    if brace_count != 0:
        errors.append(f"Mismatched braces: {brace_count} unclosed")
    return len(errors) == 0, errors


def validate_go_mod(ctx: RunContext[Any], content: str) -> tuple[bool, list[str]]:
    errors = []
    if "module " not in content:
        errors.append("Missing module declaration")
    if "go " not in content:
        errors.append("Missing Go version directive")
    return len(errors) == 0, errors


register_tool(
    "validate_protobuf", validate_protobuf, "Validate Protobuf syntax and structure", ["validation", "backend"]
)
register_tool("validate_yaml", validate_yaml, "Validate YAML syntax", ["validation", "sre", "devops"])
register_tool("validate_json", validate_json, "Validate JSON syntax", ["validation", "test"])
register_tool("parse_go_ast", parse_go_ast, "Parse Go source for structural correctness", ["validation", "backend"])
register_tool("validate_go_mod", validate_go_mod, "Validate go.mod file structure", ["validation", "backend"])
