# ODEV - AI Plugin

This plugin acts as a core library to integrate Large Language Model (LLM) capabilities across the `odev` framework. It
provides a unified, simple interface for interacting with various AI providers, allowing other plugins to easily
leverage AI for tasks like code generation, translation, and analysis.

This plugin is a foundational component and is used by other plugins such as
[`ai-scaffold`](https://github.com/odoo-odev/odev-plugin-ai-scaffold),
[`ai-upgrade`](https://github.com/odoo-odev/odev-plugin-ai-upgrade) and
[`ai-translation`](https://github.com/odoo-odev/odev-plugin-ai-translation).

## Installation

Install [odev](https://github.com/odoo-odev/odev/tree/main?tab=readme-ov-file#installation) if not already done. You'll
need odev version 4.0.0 or above.

Enable this plugin by running:

```bash
odev plugin --enable odoo-odev/odev-plugin-ai
```

This plugin depends on [`odev-plugin-project`](https://github.com/odoo-odev/odev-plugin-project), which is enabled
automatically.

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

## Commands

The plugin adds one standalone command and enhances three existing `odev` commands with an `--ai` flag. All of them
launch a CLI AI agent sandboxed with `bwrap`, pre-loaded with the relevant [skills](#skills).

| Command | Description |
| --- | --- |
| `odev ai [<database>] <prompt...>` | Launch an interactive AI CLI agent on an optional local database. Everything after the (optional) database name is taken as the prompt, so `odev ai "fix the failing test in sale"` works as written. |
| `odev run --ai [<database>]` | Run Odoo and, if the startup/installation fails, hand the logs to the agent to diagnose and fix. |
| `odev test --ai [<database>]` | Run Odoo tests and let the agent fix the failures, following the rules in the `test_skill`. |
| `odev pre-commit --ai [--run]` | Run `pre-commit` checks and let the agent fix the reported issues, then commit the fixes. `--run` runs the checks without the agent. |

### Common options

These options are shared by every AI command above (and by the commands that other AI plugins add, such as `upgrade`,
`analyze` and `scaffold`):

| Option | Description |
| --- | --- |
| `--cli <agent>` | The CLI AI agent to use: `claude`, `agy`, `copilot` or `opencode-cli`. |
| `--model <model>` | The specific model to use for the chosen agent. |
| `-y`, `--yolo` | Automatically accept all AI commands (YOLO mode). |
| `-H`, `--headless` | Run the agent non-interactively; nothing is asked on the terminal. |
| `--resume [<id>]` | Resume a previous session, by id or `latest`. Bare `--resume` resumes the latest session of the working directory. |
| `-E`, `--edit` | Open the prompt in `$EDITOR` before sending it; save it empty to abort the run. |
| `-d`, `--dirs <dirs>` | Comma-separated list of extra directories to mount read-write in the sandbox. |
| `-V`, `--version <version>` | Launch the agent in the worktree of a specific Odoo version (e.g. `17.0`, `saas-17.2`, `master`) instead of a database's repository. The worktree is created and updated automatically, and the version's own skills are loaded. |

## Skills

The [PS skills](https://github.com/odoo-ps/ps-ai-skills) are installed automatically: the repository is cloned in the
odev home directory (`skills`) and the skills a command needs are brought into the global skills directory of every
supported agent you have installed — symlinked for the agents that follow one, copied for the ones that do not
(Antigravity / `agy`). The clone is refreshed daily. Nothing is downloaded from npm.

```bash
odev config skills.disabled odoo_upgrade_skill,test_skill  # skip specific skills
odev config skills.interval 7                              # refresh the skills weekly instead of daily
```

When a run targets a specific Odoo version — named with `-V` (`odev ai -V 18.0`) or inferred from the database the agent
works on — the skills shipped with that version's source (`<worktrees>/<version>/odoo/skills`, e.g. `odoo-guidelines`)
are installed alongside the PS ones, so the agent gets the guidelines of the exact version it works on. They are swapped
out when another version is selected.

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
