<!--
Sync Impact Report
- Version change: placeholder → 1.0.0
- Modified principles:
	- [PRINCIPLE_1_NAME] → Configuration-Driven Integrations
	- [PRINCIPLE_2_NAME] → Public API Stability
	- [PRINCIPLE_3_NAME] → Tested Behavior Changes
	- [PRINCIPLE_4_NAME] → Error Clarity & Safe Logging
	- [PRINCIPLE_5_NAME] → Documentation Currency
- Added sections: Security & Configuration; Development Workflow & Quality Gates
- Removed sections: none
- Templates requiring updates:
	- .specify/templates/plan-template.md ✅ updated
	- .specify/templates/tasks-template.md ✅ updated
	- .specify/templates/spec-template.md ✅ no change required
- Follow-up TODOs: TODO(RATIFICATION_DATE): original adoption date unknown
-->
# CustomLangChain Constitution

## Core Principles

### Configuration-Driven Integrations
All model endpoints, deployments, pricing, and feature toggles MUST be sourced
from configuration (e.g., config.yaml) or environment variables. Hardcoding
service endpoints, model names, or API keys in code is forbidden. Secrets MUST
NOT be committed to the repository.

### Public API Stability
Public interfaces in langchain_openai and related modules MUST remain backward
compatible. Breaking changes require a documented migration path and a major
version bump.

### Tested Behavior Changes
Any change that alters runtime behavior MUST include corresponding test updates
or additions (unit and/or integration) that validate the new behavior.

### Error Clarity & Safe Logging
Errors surfaced to users MUST be actionable and specific. Logs MUST NOT include
secrets or sensitive tokens; redact credentials and request payloads as needed.

### Documentation Currency
Public behavior changes MUST be reflected in relevant documentation (README,
docstrings, or API reference notes) in the same change set.# From URL
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "url": "https://example.com/path/to/image.jpg"},
    ]
}

# From base64 data
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "image/jpeg",
        },
    ]
}

# From provider-managed File ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "file_id": "file-abc123"},
    ]
}

## Security & Configuration

- Secrets MUST be provided via environment variables or secure stores.
- Configuration files may include non-sensitive defaults only.
- Sensitive data MUST be excluded from logs, error messages, and test fixtures.

## Development Workflow & Quality Gates

- Changes affecting public APIs require peer review.
- Relevant unit and integration tests MUST pass before merge.
- Public-facing functions MUST retain or improve type hints.

## Governance
<!-- Example: Constitution supersedes all other practices; Amendments require documentation, approval, migration plan -->

[GOVERNANCE_RULES]
- This constitution supersedes conflicting local conventions.
- Amendments require a documented rationale, version bump, and update to the
	Sync Impact Report.
- Versioning follows semantic versioning (MAJOR.MINOR.PATCH).
- Compliance MUST be checked during planning using the Constitution Check gate
	in plan.md and during review before merge.

**Version**: 1.0.0 | **Ratified**: TODO(RATIFICATION_DATE): original adoption date unknown | **Last Amended**: 2026-02-09
<!-- Example: Version: 2.1.1 | Ratified: 2025-06-13 | Last Amended: 2025-07-16 -->
