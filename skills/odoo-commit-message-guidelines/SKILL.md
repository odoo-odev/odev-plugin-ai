---
name: odoo-commit-message-guidelines
description:
    "Draft, rewrite, and validate Odoo-style commit messages using [TAG] module: summary format, 50/72 length limits,
    imperative English, WHY-first body, and correct tag selection. Includes optional migration tagging ([MIG]) when the
    project workflow (like OCA) uses it."
metadata:
    author: "Alexander Cuellar Morales"
    version: "1.0"
---

# Odoo Commit Message Guidelines

## Purpose

Use this skill to produce high-quality Odoo commit messages that are consistent, reviewable, and easy to revert.

## Local References

-   `references/git_guidelines.md` — header heuristic and WHY-over-WHAT philosophy
-   `references/CONTRIBUTING.md` — OCA `[MIG]` tag extension and header truncation check

Consult these only for edge-case decisions or policy justification.

## When To Use

Use this skill when the user asks to:

-   write a new commit message for Odoo code
-   rewrite or improve an existing commit message
-   validate whether a commit message follows Odoo rules
-   choose the correct Odoo tag

## Required Inputs

Collect before drafting: module name (or `various`), change intent, WHY rationale, and any references (`task-*`,
`ticket-*`, `Fixes #`, `Closes #`, `opw-*`) or CI directives (`[NO CI]`).

If inputs are missing, ask only the minimum concise follow-up questions.

## Output Format

Use this structure:

```text
[TAG] module: short summary (ideally < 50 chars)

WHY the change is needed.
WHAT changed and technical choices (only if useful).

task-123
ticket-12345
Fixes #123
Closes #456
opw-789
```

Rules:

-   English only.
-   Header: `[TAG] module: imperative summary` (~50 chars).
-   Tag must be uppercase, bracketed, and from `Tag Selection Rules`.
-   Body wrapped at 72 chars; plain text (lists with `*`/`-`, no tables).
-   Imperative present voice: `Fix`, `Remove`, `Add` (not past/third-person).
-   Header must be meaningful — reject generic words like `bugfix`.
-   Prioritize WHY over WHAT in the body.
-   Tag must match change intent; one tag per header.
-   The header must not truncate with ellipsis in PR UI.
-   References (`task-*`, `Fixes #`, `Closes #`, `opw-*`) go at the end.
-   `[MIG]` is OCA-specific; use only when project convention supports it.

If any rule fails, rewrite the message before returning.

## Tag Selection Rules

Choose exactly one main tag for the commit:

-   `[FIX]` bug fix
-   `[REF]` major refactor / heavy rewrite
-   `[ADD]` add a new module
-   `[REM]` remove dead code/resources/modules
-   `[REV]` revert an earlier commit
-   `[MOV]` move files/code while preserving history intent
-   `[REL]` release commit
-   `[IMP]` incremental improvement
-   `[MERGE]` merge / forward-port integration commit
-   `[CLA]` individual contributor license signature
-   `[I18N]` translation updates
-   `[PERF]` performance improvement
-   `[CLN]` cleanup
-   `[LINT]` linting pass
-   `[MIG]` module migration (only when the project convention supports it)

Decision guidance:

-   Prefer the tag that best captures intent, not file type.
-   If it is clearly a migration, use `[MIG]`.
-   If it fixes a regression while migrating, split into two commits when possible.
-   Avoid stacking multiple tags in one header.

## Module Naming Rules

-   Use technical module name, not marketing/functional display names.
-   If multiple modules are touched, list them briefly or use `various`.
-   Prefer one logical change per commit and avoid large cross-module commits.

## Good Examples

```text
[FIX] website: remove unused alert div

Fix the look of input-group-btn.
Bootstrap requires input-group-btn to be the first or last child.
An invisible alert node broke that structure and produced visual issues.

Fixes #22769
Closes #22793
```

```text
[FIX] various: resolve rounding issues in currency conversions

Address inconsistent decimal rounding behavior across multiple reporting
and accounting modules. Instead of allowing components to do ad-hoc
rounding, enforce standard decimal precision in the core tools.

ticket-10928
[NO CI]
```

```text
[IMP] web: add module system to web client

Introduce a module system for JavaScript code to improve isolation,
load order control, and maintainability as the client grows.
```

```text
[MIG] stock_account: migrate valuation hooks to 19.0

Align valuation hook signatures with 19.0 API to preserve extension
compatibility and avoid runtime errors during upgrade.

task-8421
```

## Anti-Patterns To Reject

Reject and rewrite messages with:

-   generic headers (`bugfix`, `improvements`)
-   missing module name
-   missing rationale
-   past tense or third-person verbs in header
-   oversized multi-topic commits in one message
-   truncated first line (`...`) caused by long header

## Response Behavior

When asked to produce a commit message:

1. Return only the raw commit message (no code fences) unless the user asks for explanation.
2. If info is missing, ask only the minimum necessary clarifying questions.
3. If user provides a draft, return a corrected Odoo-compliant version.
4. Be strict: if the header does not begin with one uppercase bracketed tag from `Tag Selection Rules`, rewrite it
   before returning.
5. For IDE Source Control usage, return a ready-to-use commit message suggestion that can be inserted directly in the
   Source Control commit box (no preamble, no labels, no markdown).
6. **Execution Offer**: After providing the generated commit message, proactively offer to execute the commit on behalf
   of the user. Always ask for explicit user confirmation before running the commit. To execute safely, **write the full
   commit message to a temporary file** and run `git commit --file=<tmpfile>` (then delete the temp file). **Never
   interpolate user-provided text into shell command arguments** (`-m`) — this avoids shell metacharacter injection. Do
   not execute the commit automatically.

## Usage Examples

-   **New message:** User provides module, intent, and context → draft `[TAG] module: summary` with WHY body and
    references.
-   **Rewrite draft:** User gives a bad message like `fixed stuff` → reject, ask minimum clarifying questions, return
    corrected version.
-   **Tag choice:** User unsure between `[MIG]` and `[FIX]` → recommend based on primary intent; suggest splitting if
    both apply.
