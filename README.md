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
-   **Node.js & npm (npx)**: Essential for automatic browser provisioning (Chrome) during tour tests and AI-driven UI
    validation.
    ```bash
    # On Ubuntu/Debian
    sudo apt install nodejs npm
    ```
-   **Antigravity CLI (agy)**: Recommended CLI for Antigravity AI capabilities.
    ```bash
    curl -fsSL https://antigravity.google/cli/install.sh | bash
    ```

## Configuration

Configuration is handled automatically when you first install `odev` or enable the AI plugin. You will be prompted to
select your preferred LLM provider and enter the corresponding API key.

## Skills

The [PS skills](https://github.com/odoo-ps/ps-ai-skills) are installed automatically: the repository is cloned in the
odev home directory (`skills`) and each skill is symlinked into the global skills directory of every supported agent
you have installed. Changes are pulled once a week. Nothing is downloaded from npm.

```bash
odev config skills.disabled odoo_upgrade_skill,test_skill  # skip specific skills
odev config skills.interval 1                              # pull the skills daily instead of weekly
```

If you previously ran `npx skills add odoo-ps/ps-ai-skills`, its skills are unlinked from your agents the first time one
runs: they would otherwise shadow the git-managed ones and stay frozen at the version you installed back then. The
copies themselves are kept in `~/.agents/skills`, so any local edit you made is still there — use `npx skills remove` to
drop them for good. Skills you installed yourself and skills coming from another repository are never touched.

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
