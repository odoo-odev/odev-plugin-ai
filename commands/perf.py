"""AI-assisted performance analysis command.

Launch an Odoo database locally, let the user interactively start/stop a
profiling window (so we don't profile the whole boot/navigation, only the
operation under investigation), collect the artifacts **host-side** and feed
them to a sandboxed AI agent.

The AI never gets access to the database: odev performs every DB read
(EXPLAIN plans, ``ir.profile`` records) and writes the results to files; only
those files (and the addons source) are mounted into the AI sandbox.
"""

import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from odev.commands.database.run import RunCommand as BaseRunCommand
from odev.common import args
from odev.common.logging import logging

from odev.plugins.odev_plugin_ai.common.mixins import AICommandMixin

logger = logging.getLogger(__name__)


# Profiler choices offered to the user.
PROFILER_SQL = "sql"
PROFILER_PYSPY = "pyspy"
PROFILER_ODOO = "profiler"

# Markers used to detect that the Odoo server is ready to serve requests.
READY_RE = re.compile(
    r"HTTP service \(werkzeug\) running|Registry loaded in|Modules loaded\.|HTTP service .* running on"
)
# A line emitted by the ``odoo.sql_db`` logger when ``--log-handler=odoo.sql_db:DEBUG`` is set.
SQL_LINE_RE = re.compile(r"odoo\.sql_db:\s*(?P<body>.*)$")
# Best-effort extraction of a per-query duration (ms) when the version logs it.
SQL_DURATION_RE = re.compile(r"\[?\s*(?P<ms>\d+(?:\.\d+)?)\s*ms\]?", re.IGNORECASE)

# Sentinels printed by the odoo-bin shell snippets so we can reliably parse stdout
# (which is otherwise polluted by Odoo's own startup logs).
SENTINEL_START = "___ODEV_PERF_START___"
SENTINEL_PROFILE = "___ODEV_PERF_PROFILE___"
SENTINEL_DONE = "___ODEV_PERF_DONE___"


class PerfCommand(BaseRunCommand, AICommandMixin):
    """Profile a running Odoo database and let an AI agent analyze the results.

    Launches the database, lets you choose which profilers to enable (SQL query
    logging, py-spy CPU sampling, Odoo's internal profiler), then drive a
    start/stop window around the slow operation. Collected artifacts are handed
    to a sandboxed AI agent **without** any database access.
    """

    _name = "perf"

    explain_top = args.Integer(
        name="explain_top",
        aliases=["--explain-top"],
        description="Number of slowest SQL queries to run EXPLAIN on (default: 10).",
        default=10,
    )
    explain_analyze = args.Flag(
        name="explain_analyze",
        aliases=["--explain-analyze"],
        description="Use EXPLAIN (ANALYZE, BUFFERS) instead of plain EXPLAIN. "
        "Only applied to SELECT queries (it executes the query).",
        default=False,
    )

    # --------------------------------------------------------------------------
    # Entry point
    # --------------------------------------------------------------------------

    def run(self):
        selected = self._select_profilers()
        if not selected:
            logger.warning("No profiler selected, nothing to do.")
            return

        odoo_args = self._prepare_odoo_args(selected)
        session_dir = self._make_session_dir()
        logfile = session_dir / "odoo.log"

        # Copy previous sessions history to session_dir
        self._copy_previous_sessions_history(session_dir)

        proc = None
        log_handle = None
        pyspy_proc = None
        profiler_start_id = None
        sql_start_offset = None
        odoo_start_offset = None

        try:
            log_handle = open(logfile, "wb")  # noqa: SIM115 - kept open for the whole run
            proc = self._launch_odoo(odoo_args, log_handle)
            self._wait_until_ready(proc, logfile, odoo_args)

            if PROFILER_ODOO in selected:
                profiler_start_id = self._enable_odoo_profiler()
                logger.info("IMPORTANT: To record Odoo traces, you MUST append '?profile=1' to your browser URL right now.")

            if not self.console.confirm(
                "Navigate to just before the slow operation (add ?profile=1 to URL if needed), then confirm to START",
                default=True,
            ):
                logger.warning("Profiling aborted by user.")
                return

            # --- START -------------------------------------------------------
            odoo_start_offset = logfile.stat().st_size
            if PROFILER_SQL in selected:
                sql_start_offset = logfile.stat().st_size
            if PROFILER_PYSPY in selected:
                pyspy_proc = self._start_pyspy(proc, session_dir)

            logger.info("Profiling started. Reproduce the slow operation now.")

            while not self.console.confirm("Done? Confirm to STOP profiling and collect results", default=True):
                continue

            # --- STOP / collect ---------------------------------------------
            if pyspy_proc is not None:
                self._stop_pyspy(pyspy_proc, session_dir)
                pyspy_proc = None
            if PROFILER_SQL in selected and sql_start_offset is not None:
                self._collect_sql(logfile, sql_start_offset, session_dir)
            if PROFILER_ODOO in selected and profiler_start_id is not None:
                self._dump_odoo_profiler(profiler_start_id, session_dir)
                profiler_start_id = None

        finally:
            # Always clean up: kill py-spy, disable Odoo profiling, stop Odoo.
            if pyspy_proc is not None and pyspy_proc.poll() is None:
                self._terminate(pyspy_proc, signal.SIGINT)
            if PROFILER_ODOO in selected and profiler_start_id is not None:
                self._disable_odoo_profiler()
            if proc is not None:
                self._stop_odoo(proc)
            if log_handle is not None:
                log_handle.close()

            # Slice and save odoo.log for this run
            if odoo_start_offset is not None and logfile.exists():
                self._collect_odoo_log(logfile, odoo_start_offset, session_dir)

        self._launch_ai(session_dir, selected)

    # --------------------------------------------------------------------------
    # 1. Profiler selection & odoo args
    # --------------------------------------------------------------------------

    @staticmethod
    def _get_pyspy_path() -> str | None:
        """Find py-spy in PATH or the current virtual environment."""
        path = shutil.which("py-spy")
        if path:
            return path
        venv_path = os.path.join(os.path.dirname(sys.executable), "py-spy")
        if os.path.isfile(venv_path) and os.access(venv_path, os.X_OK):
            return venv_path
        return None

    def _select_profilers(self) -> list[str]:
        """Prompt the user for the performance aspects to profile."""
        logger.info(
            f"About to launch Odoo on database {self.database_name!r}. "
            "Choose which performance aspects you want to analyze."
        )
        choices = [
            (PROFILER_SQL, "SQL queries (debug_sql) + EXPLAIN on the slowest"),
            (PROFILER_PYSPY, "CPU sampling (py-spy flamegraph)"),
            (PROFILER_ODOO, "Odoo internal profiler (ir.profile)"),
        ]
        selected: list[str] = (
            self.console.checkbox(
                "Performance aspects to profile",
                choices=choices,
                defaults=[PROFILER_PYSPY],
            )
            or []
        )

        if PROFILER_PYSPY in selected and not self._get_pyspy_path():
            logger.warning(
                "py-spy is not installed (not found in PATH). Skipping CPU sampling. "
                "Install it with 'cargo install py-spy' or 'pip install py-spy'."
            )
            selected = [s for s in selected if s != PROFILER_PYSPY]
        elif PROFILER_PYSPY in selected:
            try:
                with open("/proc/sys/kernel/yama/ptrace_scope", "r") as f:
                    if f.read().strip() != "0":
                        logger.warning(
                            "kernel.yama.ptrace_scope is not 0. py-spy might fail to attach. "
                            "Consider running 'sudo sysctl kernel.yama.ptrace_scope=0' in another terminal."
                        )
            except FileNotFoundError:
                pass

        return selected

    def _prepare_odoo_args(self, selected: list[str]) -> list[str]:
        """Adapt the odoo-bin arguments to the selected profilers."""
        odoo_args = list(self.args.odoo_args or [])

        # A live server is required: drop any stop-after-init.
        stripped = [a for a in odoo_args if a not in ("--stop-after-init", "--st")]
        if len(stripped) != len(odoo_args):
            logger.warning("Removed --stop-after-init: a running server is required for profiling.")
            odoo_args = stripped

        if PROFILER_SQL in selected and not any("odoo.sql_db" in a for a in odoo_args):
            odoo_args.append("--log-handler=odoo.sql_db:DEBUG")

        if PROFILER_PYSPY in selected and not any(a.startswith("--workers") for a in odoo_args):
            # Threaded mode: web requests are served by the main process, the one we sample.
            odoo_args.append("--workers=0")

        return odoo_args

    # --------------------------------------------------------------------------
    # 2. Launch Odoo in the background (controllable)
    # --------------------------------------------------------------------------

    def _launch_odoo(self, odoo_args: list[str], log_handle) -> subprocess.Popen:
        """Launch odoo-bin as a background process writing to the log file."""
        if self.odoobin is None:
            raise self.error(f"Could not spawn process for database {self.database_name!r}")

        if not self.odoobin.venv.exists:
            self.odoobin.prepare_odoobin()

        full_args = self.odoobin.prepare_odoobin_args(odoo_args)
        command = [str(self.odoobin.venv.python), str(self.odoobin.odoobin_path), *full_args]
        logger.debug(f"Launching Odoo: {' '.join(command)}")

        return subprocess.Popen(  # noqa: S603 - args built from trusted odev internals
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _wait_until_ready(self, proc: subprocess.Popen, logfile: Path, odoo_args: list[str], timeout: int = 180):
        """Poll the log file until Odoo is ready (or it exits / times out)."""
        port = self._http_port(odoo_args)
        logger.info(f"Starting Odoo... (logs: {logfile})")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                tail = self._read_tail(logfile)
                raise self.error(f"Odoo exited before becoming ready (code {proc.returncode}):\n{tail}")

            try:
                content = logfile.read_text(errors="replace")
            except OSError:
                content = ""

            if READY_RE.search(content):
                time.sleep(1)  # let the HTTP socket finish binding
                logger.info(f"Odoo is running on http://localhost:{port}/web")
                logger.info(f"You can follow the logs with: tail -f {logfile}")
                return

            time.sleep(0.5)

        raise self.error(f"Odoo did not become ready within {timeout}s (logs: {logfile})")

    @staticmethod
    def _http_port(odoo_args: list[str]) -> int:
        """Extract the HTTP port from odoo args, defaulting to 8069."""
        for i, arg in enumerate(odoo_args):
            match = re.match(r"(?:-p|--http-port)(?:=(\d+))?$", arg)
            if match:
                if match.group(1):
                    return int(match.group(1))
                if i + 1 < len(odoo_args) and odoo_args[i + 1].isdigit():
                    return int(odoo_args[i + 1])
        return 8069

    @staticmethod
    def _read_tail(logfile: Path, lines: int = 40) -> str:
        try:
            return "\n".join(logfile.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return "<no log available>"

    # --------------------------------------------------------------------------
    # 3. py-spy
    # --------------------------------------------------------------------------

    def _start_pyspy(self, proc: subprocess.Popen, session_dir: Path) -> subprocess.Popen | None:
        """Spawn py-spy against the running Odoo process (host-side)."""
        pid = self.odoobin.pid or proc.pid
        output = session_dir / "pyspy.folded"
        pyspy_bin = self._get_pyspy_path() or "py-spy"
        command = [
            pyspy_bin,
            "record",
            "--pid",
            str(pid),
            "--format",
            "raw",
            "--subprocesses",
            "--output",
            str(output),
        ]
        logger.info(f"Attaching py-spy to PID {pid} (output: {output.name})")
        try:
            pyspy_proc = subprocess.Popen(command)  # noqa: S603
        except OSError as error:
            logger.error(f"Failed to start py-spy: {error}")
            return None

        # py-spy fails fast on a permission error; surface a helpful message.
        time.sleep(1)
        if pyspy_proc.poll() is not None and pyspy_proc.returncode:
            logger.error(
                "py-spy could not attach (permission denied?). On Linux you may need to run with "
                "elevated privileges (sudo py-spy ...) or relax ptrace: "
                "'sudo sysctl kernel.yama.ptrace_scope=0'."
            )
            return None
        return pyspy_proc

    def _stop_pyspy(self, pyspy_proc: subprocess.Popen, session_dir: Path):
        """Stop py-spy so it flushes the collapsed-stacks file."""
        logger.info("Stopping py-spy and saving flamegraph data...")
        self._terminate(pyspy_proc, signal.SIGINT)
        output = session_dir / "pyspy.folded"
        if not output.exists() or output.stat().st_size == 0:
            logger.warning("py-spy did not produce any samples (operation too short or attach failed).")

    # --------------------------------------------------------------------------
    # 4. SQL window + EXPLAIN
    # --------------------------------------------------------------------------

    def _collect_sql(self, logfile: Path, start_offset: int, session_dir: Path):
        """Slice the log between start/stop, save SQL queries and EXPLAIN the slowest."""
        with open(logfile, "rb") as handle:
            handle.seek(start_offset)
            window = handle.read().decode(errors="replace")

        sql_lines = [m.group(0) for m in (SQL_LINE_RE.search(line) for line in window.splitlines()) if m]
        queries_file = session_dir / "sql_queries.log"
        queries_file.write_text("\n".join(sql_lines))
        logger.info(f"Saved {len(sql_lines)} SQL log lines to {queries_file.name}")

        ranked = self._rank_queries(window)
        if not ranked:
            logger.warning("No SQL queries captured during the window (nothing to EXPLAIN).")
            return

        self._explain_queries(ranked[: self.args.explain_top], session_dir)

    def _collect_odoo_log(self, logfile: Path, start_offset: int, iter_dir: Path):
        """Slice the log between start/stop and save it to the iteration folder."""
        try:
            with open(logfile, "rb") as handle:
                handle.seek(start_offset)
                window = handle.read().decode(errors="replace")
            (iter_dir / "odoo.log").write_text(window)
            logger.info(f"Saved iteration-specific Odoo log to {iter_dir.name}/odoo.log")
        except Exception as error:
            logger.warning(f"Failed to slice odoo.log for iteration: {error}")

    def _rank_queries(self, window: str) -> list[tuple[float, int, str]]:
        """Return queries ranked by (measured time desc, frequency desc).

        Each item is ``(max_ms, count, query)``. When the Odoo version does not
        log per-query timings, ``max_ms`` is 0 and ranking falls back to
        frequency (which surfaces N+1 query patterns).
        """
        stats: dict[str, list[float]] = {}
        for line in window.splitlines():
            match = SQL_LINE_RE.search(line)
            if not match:
                continue
            body = match.group("body").strip()
            duration = SQL_DURATION_RE.search(body)
            ms = float(duration.group("ms")) if duration else 0.0
            # Strip a leading "query:" / "[x ms]" prefix to keep the bare statement.
            query = SQL_DURATION_RE.sub("", body).strip()
            query = re.sub(r"^query:\s*", "", query).strip()
            if not query:
                continue
            stats.setdefault(query, []).append(ms)

        ranked = [(max(times), len(times), query) for query, times in stats.items()]
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return ranked

    def _explain_queries(self, ranked: list[tuple[float, int, str]], session_dir: Path):
        """Run EXPLAIN on the given queries via odev's DB access (not the AI's)."""
        analyze = self.args.explain_analyze
        out_lines: list[str] = []
        explained = 0

        for max_ms, count, query in ranked:
            prefix = "EXPLAIN (ANALYZE, BUFFERS) " if (analyze and query.lower().startswith("select")) else "EXPLAIN "
            header = f"-- max {max_ms:.3f} ms, executed {count}x\n{query}"
            try:
                result = self._database.query(prefix + query)
                plan = "\n".join(row[0] for row in (result or []))
                out_lines.append(f"{header}\n{prefix.strip()}:\n{plan}\n")
                explained += 1
            except Exception as error:  # noqa: BLE001 - log line may not be executable (placeholders, truncation)
                out_lines.append(f"{header}\n-- EXPLAIN skipped: {error}\n")

        explain_file = session_dir / "sql_explain_plans.txt"
        explain_file.write_text("\n".join(out_lines))
        logger.info(f"Wrote EXPLAIN plans for {explained}/{len(ranked)} slow queries to {explain_file.name}")

    # --------------------------------------------------------------------------
    # 5. Odoo internal profiler (via odoo-bin shell, server-side)
    # --------------------------------------------------------------------------

    def _enable_odoo_profiler(self) -> int | None:
        """Enable Odoo profiling in the running server; return the current max ir.profile id."""
        code = (
            "import json\n"
            "from datetime import datetime, timedelta\n"
            "until = (datetime.utcnow() + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')\n"
            "env['ir.config_parameter'].sudo().set_param('base.profiling_enabled_until', until)\n"
            "env.cr.commit()\n"
            "last = env['ir.profile'].search([], order='id desc', limit=1)\n"
            f"print({SENTINEL_START!r} + json.dumps({{'start_id': last.id if last else 0}}))\n"
        )
        stdout = self._run_shell(code)
        for line in (stdout or "").splitlines():
            if SENTINEL_START in line:
                payload = json.loads(line.split(SENTINEL_START, 1)[1])
                logger.info("Odoo internal profiler enabled.")
                return payload.get("start_id", 0)

        logger.warning("Could not enable the Odoo internal profiler (ir.profile may be unavailable on this version).")
        return None

    def _dump_odoo_profiler(self, start_id: int, session_dir: Path):
        """Export ir.profile records created since start_id and disable profiling."""
        code = (
            "import json\n"
            f"profiles = env['ir.profile'].search([('id', '>', {int(start_id)})])\n"
            "fields_to_dump = ['id','name','duration','entry_count','sql','sql_count',"
            "'init_stack_trace','traces_async','traces_sync','create_date']\n"
            "for p in profiles:\n"
            "    rec = {}\n"
            "    for f in fields_to_dump:\n"
            "        try:\n"
            "            rec[f] = p[f]\n"
            "        except Exception:\n"
            "            rec[f] = None\n"
            f"    print({SENTINEL_PROFILE!r} + json.dumps(rec, default=str))\n"
            "env['ir.config_parameter'].sudo().set_param('base.profiling_enabled_until', '0')\n"
            "env.cr.commit()\n"
            f"print({SENTINEL_DONE!r})\n"
        )
        stdout = self._run_shell(code)
        records = []
        for line in (stdout or "").splitlines():
            if SENTINEL_PROFILE in line:
                try:
                    records.append(json.loads(line.split(SENTINEL_PROFILE, 1)[1]))
                except json.JSONDecodeError:
                    continue

        out_file = session_dir / "odoo_profiler.json"
        out_file.write_text(json.dumps(records, indent=2))
        logger.info(f"Exported {len(records)} ir.profile record(s) to {out_file.name}")

    def _disable_odoo_profiler(self):
        """Best-effort: make sure profiling is turned off even on error."""
        code = "env['ir.config_parameter'].sudo().set_param('base.profiling_enabled_until', '0')\n" "env.cr.commit()\n"
        try:
            self._run_shell(code)
        except Exception:  # noqa: BLE001 - cleanup must never raise
            logger.debug("Could not disable Odoo profiling during cleanup.", exc_info=True)

    def _run_shell(self, code: str) -> str | None:
        """Run a python snippet inside an odoo-bin shell and return its stdout."""
        process = self.odoobin.run(subcommand="shell", subcommand_input=code, stream=False)
        if process is None or process.stdout is None:
            return None
        return process.stdout.decode(errors="replace")

    # --------------------------------------------------------------------------
    # 6. Process shutdown
    # --------------------------------------------------------------------------

    def _stop_odoo(self, proc: subprocess.Popen):
        """Stop the Odoo server gracefully, escalating if needed."""
        if proc.poll() is not None:
            return
        logger.info("Stopping Odoo...")
        self._terminate(proc, signal.SIGINT, timeout=30)

    @staticmethod
    def _terminate(proc: subprocess.Popen, sig: int, timeout: int = 15):
        """Signal a process group and wait, escalating to SIGKILL on timeout."""
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    # --------------------------------------------------------------------------
    # 7. Launch the AI on the collected artifacts (no DB)
    # --------------------------------------------------------------------------

    def _make_session_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_dir = Path.cwd() / ".odev-perf" / f"{self.database_name}-{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _find_previous_findings(self, session_dir: Path) -> Path | None:
        """Return the most recent PERF_FINDINGS.md from a previous session, if any."""
        perf_root = session_dir.parent
        candidates = sorted(
            perf_root.glob(f"{self.database_name}-*/PERF_FINDINGS.md"),
            key=lambda p: p.parent.name,
            reverse=True,
        )
        # Skip the current session (it has no findings yet).
        return next((p for p in candidates if p.parent != session_dir), None)

    def _copy_previous_sessions_history(self, session_dir: Path, limit: int = 5):
        """Locate up to `limit` previous sessions and copy their key artifacts to a history subfolder."""
        perf_root = session_dir.parent
        session_dirs = sorted(
            [d for d in perf_root.glob(f"{self.database_name}-*") if d.is_dir() and d != session_dir],
            key=lambda d: d.name,
            reverse=True,
        )

        for prev_dir in session_dirs[:limit]:
            dest_dir = session_dir / "previous_attempts" / prev_dir.name
            root_key_files = [
                "PERF_FINDINGS.md",
                "PERF_ANALYSIS.md",
                "pyspy.folded",
                "sql_queries.log",
                "sql_explain_plans.txt",
                "odoo_profiler.json",
                "odoo.log",
            ]

            for f_name in root_key_files:
                src_file = prev_dir / f_name
                if src_file.exists():
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dest_dir / f_name)

            for iter_dir in prev_dir.glob("iteration_*"):
                if iter_dir.is_dir():
                    for f_name in root_key_files:
                        src_file = iter_dir / f_name
                        if src_file.exists():
                            iter_dest = dest_dir / iter_dir.name
                            iter_dest.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_file, iter_dest / f_name)

    def _write_index(self, session_dir: Path, selected: list[str]) -> Path:
        """Write a human/AI-readable index describing the captured artifacts."""
        lines = [
            "# Performance analysis artifacts",
            "",
            f"- Database: `{self.database_name}`",
            f"- Odoo version: `{self.version}`",
            f"- Profilers: {', '.join(selected)}",
            "",
            "## Current Profiling Files",
            "The following profiling artifacts have been recorded in this session:",
        ]

        if PROFILER_PYSPY in selected:
            lines.append("- `pyspy.folded`: py-spy collapsed stacks (`frame;frame;... <samples>`). The most-sampled stacks are the CPU hot paths.")
        if PROFILER_SQL in selected:
            lines.append("- `sql_queries.log`: raw SQL queries logged during the profiling window.")
            lines.append("- `sql_explain_plans.txt`: PostgreSQL EXPLAIN plans for the slowest/most-frequent queries.")
        if PROFILER_ODOO in selected:
            lines.append("- `odoo_profiler.json`: Odoo `ir.profile` records (speedscope traces, SQL, durations).")
        lines.append("- `odoo.log`: sliced Odoo server log containing only the profiling window execution.")
        lines.append("")

        history_dir = session_dir / "previous_attempts"
        if history_dir.exists() and any(history_dir.iterdir()):
            lines.extend([
                "## Previous Attempts History",
                "Artifacts from previous runs are available in `previous_attempts/`:",
                "",
            ])
            for prev in sorted(history_dir.iterdir(), key=lambda p: p.name, reverse=True):
                lines.append(f"### Session: `{prev.name}`")
                if (prev / "PERF_FINDINGS.md").exists():
                    lines.append(f"- [PERF_FINDINGS.md](previous_attempts/{prev.name}/PERF_FINDINGS.md): Findings/result of investigation from this attempt.")
                if (prev / "pyspy.folded").exists():
                    lines.append("- `pyspy.folded`: CPU sampling raw data (folded stacks).")
                if (prev / "sql_queries.log").exists():
                    lines.append("- `sql_queries.log`: Raw SQL query log.")
                    lines.append("- `sql_explain_plans.txt`: SQL explain plans.")
                if (prev / "odoo_profiler.json").exists():
                    lines.append("- `odoo_profiler.json`: Odoo ir.profile json records.")
                if (prev / "odoo.log").exists():
                    lines.append("- `odoo.log`: Sliced Odoo server log.")
                for sub in sorted(prev.glob("iteration_*"), key=lambda p: int(p.name.split("_")[1])):
                    lines.append(f"- `{sub.name}/`: Profiling stats for {sub.name}.")
                lines.append("")

        index = session_dir / "PERF_ANALYSIS.md"
        index.write_text("\n".join(lines) + "\n")
        return index

    def _launch_ai(self, session_dir: Path, selected: list[str]):
        """Hand the artifacts to a sandboxed AI agent, without any DB access."""
        self._write_index(session_dir, selected)

        # Mount the addons source (RW, so the AI can apply fixes) and the artifacts dir.
        addons = [str(p) for p in (self.odoobin.additional_addons_paths or [])]
        self.args.dirs = list(self.args.dirs or []) + addons + [str(session_dir)]

        # --- Previous findings (optional context for verification runs) ---
        previous_findings = self._find_previous_findings(session_dir)
        if previous_findings:
            logger.info(f"Previous findings found: {previous_findings} — the AI will compare against them.")
            previous_context = (
                "\n\n## Previous session findings\n"
                f"A previous profiling session was run on this database. "
                f"The AI wrote the following findings and applied fixes (file: `{previous_findings}`):\n\n"
                f"{previous_findings.read_text(errors='replace')}\n"
                "---\n"
                "Your primary goal this session is to **verify** whether those fixes resolved the reported "
                "bottlenecks. Compare the new artifacts against the stats reported above. Clearly state for "
                "each previous finding: FIXED, IMPROVED, or STILL PRESENT (with new evidence)."
            )
        else:
            previous_context = ""

        # --- History folder context ---
        history_dir = session_dir / "previous_attempts"
        history_context = ""
        if history_dir.exists() and any(history_dir.iterdir()):
            history_context = (
                "\n\n## Previous Attempts History\n"
                "You can find raw statistics and findings from previous profiling sessions "
                f"in the `previous_attempts/` folder. Use these folders to do direct comparisons "
                "with past profiling iterations (e.g. comparing past pyspy.folded or SQL query logs) "
                "to verify improvements quantitatively."
            )

        # --- Findings template the AI must fill in ---
        findings_file = session_dir / "PERF_FINDINGS.md"
        findings_instructions = (
            f"\n\n## Required output: save findings to `{findings_file}`\n"
            "At the END of your analysis, you MUST write a `PERF_FINDINGS.md` file "
            f"in `{session_dir}` with the following sections:\n"
            "```\n"
            "# Perf findings — <database> — <timestamp>\n"
            "## Key stats\n"
            "(Total SQL queries, distinct queries, slowest query ms, py-spy top-3 frames, etc.)\n"
            "## Bottlenecks\n"
            "(One entry per bottleneck: description, artifact+evidence, severity)\n"
            "## Fixes applied\n"
            "(File changed, what was changed, why)\n"
            "## Recommendations (not yet applied)\n"
            "(e.g. DB migrations, index additions, config changes)\n"
            "```\n"
            "This file is used by the NEXT profiling run to verify that your fixes worked."
        )

        prompt = (
            "You are analyzing the performance of an Odoo custom module.\n\n"
            f"All profiling data has been collected for you in `{session_dir}` "
            f"(see `{session_dir / 'PERF_ANALYSIS.md'}` for the index of artifacts). "
            "You have NO database access: everything you need is in those files."
            f"{previous_context}{history_context}\n\n"
            "Please:\n"
            "1. Read PERF_ANALYSIS.md and inspect every artifact.\n"
            "2. Identify the concrete performance bottlenecks (CPU hot paths from py-spy, "
            "slow/N+1 SQL queries and their EXPLAIN plans, expensive ORM traces from the Odoo profiler).\n"
            "3. Apply optimizations IN THE CUSTOM MODULE code (the addons paths are mounted read-write): "
            "e.g. batch ORM calls, add prefetch/read_group, fix N+1 loops, add missing indexes, "
            "cache computed values. Do NOT modify Odoo core.\n"
            "4. Finish with a clear summary: each bottleneck found, the evidence (which artifact), "
            "and the change you made (or recommend, if it requires a migration/index).\n"
            f"{findings_instructions}"
        )

        self.run_ai_agent(prompt, database=None, ephemeral_pg=False)
