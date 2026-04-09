# ODEV - AI Plugin

This plugin acts as a core library to integrate Large Language Model (LLM) capabilities across the `odev` framework. It
provides a unified, simple interface for interacting with various AI providers, allowing other plugins to easily
leverage AI for tasks like code generation, translation, and analysis.

This plugin is a foundational component and is used by other plugins such as `scaffold` and `translate`.

## Prerequisites

For the AI agent to work correctly (especially for `odev pre-commit --ai`), the following tools must be installed on
your host system:

-   **pre-commit**: Required to run and verify linting checks.
    ```bash
    pip install pre-commit
    ```
-   **ruff**: A fast linter used by many Odoo projects.
    ```bash
    pip install ruff
    ```
-   **RTK (Rust Token Killer)**: Highly recommended for token optimization (saves 60-90% on command outputs). `odev`
    automatically binds your host's `rtk` binary and its cache (`~/.cache/rtk`) into the AI sandbox.
    ```bash
    curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
    rtk init --global
    ```

## Configuration

Configuration is handled automatically when you first install `odev` or enable the AI plugin. You will be prompted to
select your preferred LLM provider and enter the corresponding API key.
