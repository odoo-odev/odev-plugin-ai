# ODEV - AI Plugin

This plugin acts as a core library to integrate Large Language Model (LLM) capabilities across the `odev` framework. It
provides a unified, simple interface for interacting with various AI providers, allowing other plugins to easily
leverage AI for tasks like code generation, translation, and analysis.

This plugin is a foundational component and is used by other plugins such as `scaffold` and `translate`.

## Prerequisites

For the AI agent to work correctly and efficiently across all tasks (testing, upgrading, pre-commit fixes, etc.), the
following tools must be installed on your host system:

-   **pre-commit**: Essential for the AI to run and verify linting checks automatically.
    ```bash
    pip install pre-commit
    ```
-   **ruff**: A primary linter used by the AI agent to analyze and fix code style and quality issues.
    ```bash
    pip install ruff
    ```
-   **RTK (Rust Token Killer)**: **Critical for performance and cost.** It optimizes almost all command outputs sent to
    the AI (saving 60-90% on token usage). `odev` automatically binds your host's `rtk` binary and its cache
    (`~/.cache/rtk`) into the AI sandbox.
    ```bash
    curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
    rtk init --global
    ```

## Configuration

Configuration is handled automatically when you first install `odev` or enable the AI plugin. You will be prompted to
select your preferred LLM provider and enter the corresponding API key.

### Using a separate Claude account for `odev ai`

`odev ai` runs the `claude` CLI against your normal Claude login by default. To bill its usage to a _different_ Claude
subscription, give it an isolated config directory:

1. Log the second account into its own dir (once), using an absolute path:

    ```bash
    CLAUDE_CONFIG_DIR="$HOME/.claude-odev" claude   # then /login as the 2nd account
    ```

2. Point odev at it:

    ```bash
    odev config ai.claude_config_dir .claude-odev
    ```

3. `odev ai` now uses that account; your normal Claude Code is untouched.

Per-invocation override: `ODEV_CLAUDE_CONFIG_DIR=.claude-other odev ai "..."`. Revert to the default account:
`odev config ai.claude_config_dir ""`.

Notes: requires `claude` ≥ 2.1.56 (namespaced macOS Keychain). The path string must match between the one-time `/login`
and odev. On some macOS setups a re-`/login` may be needed after reboot (Claude Code Keychain ACL behavior).

## RTK (Rust Token Killer) Integration

`odev` now automatically supports [RTK](https://github.com/rtk-ai/rtk) to compress terminal output and reduce LLM token
consumption.

### Installation

If you don't have RTK installed, you can install it using the official script:

```bash
curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
```

### Setup for Claude Code

To benefit from transparent command rewriting in Claude Code, run:

```bash
rtk init --global
```

This will configure a `PreToolUse` hook that automatically wraps your commands with `rtk`.
