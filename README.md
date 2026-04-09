# ODEV - AI Plugin

This plugin acts as a core library to integrate Large Language Model (LLM) capabilities across the `odev` framework. It
provides a unified, simple interface for interacting with various AI providers, allowing other plugins to easily
leverage AI for tasks like code generation, translation, and analysis.

This plugin is a foundational component and is used by other plugins such as `scaffold` and `translate`.

## Configuration

Configuration is handled automatically when you first install `odev` or enable the AI plugin. You will be prompted to
select your preferred LLM provider and enter the corresponding API key.

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
