---
name: odev
description:
    "Mandatory CLI usage rules for Odoo development, covering database creation, versioning, argument ordering, and
    server orchestration with odev."
---

# ODEV Commands Skill

This skill provides instructions on how to use `odev` (Odoo Development CLI) during upgrade tasks.

## 🛑 MANDATORY RULES (ODEV CLI USAGE)

1. **VERSIONING**: You MUST ALWAYS specify the Odoo version explicitly using the `-V <version>` flag for ALL `odev`
   commands (like `odev run`, `odev create`, `odev test`). If you omit `-V`, `odev` defaults to the `master` version.
   While this might be correct if you are targeting master, it is HIGHLY RECOMMENDED to always specify it to ensure the
   correct environment and dependencies are used.

    - **Example**: `odev run -V 19.0 ...`

2. **ARGUMENT ORDER**: `odev` specific flags (LIKE `-V`, `-f`, `-c`, `-w`, `--venv`) MUST come BEFORE the database name.

3. **DATABASE CREATION/OVERWRITE**: You MUST ALWAYS use the `-f` (force) flag when creating a database that might
   already exist to bypass confirmation prompts. "Overwriting" an existing database via `odev create` REQUIRES `-f`.

    - **Example**: `odev create -f -V 18.0 new_db -i base`

4. **LOG LEVEL**: Use `--log-level=warn` to keep logs concise and save tokens. Note that `odev` consumes this flag to
   set BOTH its own log level and the Odoo server log level.

5. **GLOB QUOTING**: When using `--glob`, you MUST ALWAYS wrap the pattern in double quotes to prevent the shell from
   expanding it before it reaches `odev`.
    - **Incorrect**: `odev upgrade-code --glob my_module/**/*`
    - **MANDATORY**: You MUST always include `--stop-after-init` when using `odev run` or `odev deploy` for
      verification. Failure to do so will cause the shell to hang.
6. **ADDONS PATHS**: `odev` automatically manages Odoo and Enterprise addons paths based on the specified version
   (`-V`). You should **NEVER** manually specify standard Odoo paths (like `odoo/addons` or `enterprise`). `odev`
   handles this internally. Additionally, `odev` automatically includes the repository path of the targeted database. If
   you are working in a custom folder, `odev` will often include it if it's detected as an addons path. Trust `odev` for
   the plumbing to avoid version mismatches.

## Core Commands

-   **`odev run <database> [options]`**:

    -   Starts the Odoo server.
    -   Useful for manual verification and testing.
    -   **MANDATORY**: You MUST always specify the Odoo version with `-V <version>` BEFORE the database name.
    -   **MANDATORY**: Always specify `--stop-after-init` when running in a script or for automated verification.
    -   **Best Practice**: Always specify `--http-port <free_port>` to avoid conflicts.
    -   **Example**: `odev run -V 19.0 db -i mod --http-port 8069 --log-level=warn --stop-after-init`

-   **`odev create -V <version> <database> [options]`**:

    -   Creates a new database and installs modules.
    -   **MANDATORY**: You MUST always specify the Odoo version with `-V <version>`.
    -   **MANDATORY**: Use `-f` to force creation if the DB already exists.
    -   Use `-T` to create a template database (e.g., `odev create -f -T -V 17.0 my_template -i sale`).
    -   Use `-t <template>` to clone from a template (e.g., `odev create -f -t my_template -V 17.0 new_db`).

-   **`odev test <database> -i <modules> [options]`**:

    -   **SURGICAL VERIFICATION**: You are ENCOURAGED to use this command to verify individual fixes, but ONLY with
        specific tags to keep it fast.
    -   **MANDATORY**: You MUST use `-t <tag>` or `--tags <tag>` to run ONLY the relevant failing test.
    -   **Example**: `odev test -V 19.0 my_db -i my_module -t .test_some_method`
    -   **Avoid Full Suites**: DO NOT run `odev test` without specific tags. Full module tests take too long and consume
        too many tokens. The host framework will run the full suite automatically after your session.

-   **`odev venv <database> -c "<command>"`**:

    -   Runs a command inside the virtual environment associated with the database.
    -   **Package Installation**: If a python package is missing, use:
        `odev venv <database> -c "pip install <package>"`.

-   **`odev shell <database> [options]`**:

    -   Starts an interactive Odoo shell for the specified database.
    -   Use `--script "<python_code>"` to execute code and exit immediately.
    -   **Example**: `odev shell my_db --script "print(self.env.user.name)"`

-   **`odev deploy <module_path> [options]`**:

    -   Hot-deploys a module to a **running** Odoo instance.
    -   **Example**: `odev deploy /custom/my_module`

-   **`odev kill -H <database> [options]`**:

    -   Kills the running Odoo process associated with a database.
    -   Use `-H` (hard) to send a SIGKILL if the process is stuck.
    -   **Example**: `odev kill my_db`

-   **`odev delete -f <database> [options]`**:

    -   Deletes an existing database and its associated resources (filestore, config).
    -   **MANDATORY**: Use `-f` (force) to bypass confirmation prompts.
    -   **Prerequisite**: If the database is currently running, you MUST kill it first using `odev kill <database>`.
    -   **Example**: `odev delete -f my_db`

-   **`odev upgrade-code <database> --from <ver> --to <ver>`**:
    -   Automatically migrates source code for common renames (e.g., `<tree>` to `<list>` in Odoo 18.0+).
    -   **MANDATORY**: You MUST specify the **target database name**, not a directory path.

## Best Practices

-   **Speed up iteration**: Use template databases (`-T` and `-t`) to avoid reinstalling standard Odoo modules
    repeatedly.
-   **Strict Versioning**: Always pass the correct `-V` flag to ensure the right Odoo worktree and virtual environment
    are used.
-   **Prompt Bypassing**: Always use `-f` in scripts or automated tasks to ensure no interactive prompts block
    execution.

## 📦 MODULE MANIFEST STANDARDS

1. **VERSIONING**: When upgrading a module to a new Odoo version, you MUST reset the version in `__manifest__.py` to
   `ODOO_VERSION.1.0.0` (e.g., `19.0.1.0.0`).
    - **NEVER** preserve minor versions or patch increments from the source Odoo version (e.g., do not turn `16.3.1.2.3`
      into `19.3.1.2.3`).
    - The format MUST be: `<odoo_major>.0.1.0.0` for the first upgrade commit.
