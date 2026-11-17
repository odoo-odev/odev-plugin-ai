"""Export the Excalidraw diagrams a task links to, as SVGs the agent can read.

A diagram is a live document: there is no url that hands back a picture, so the only
way to get one is to open the board in a browser and drive its own export. That is what
Playwright is here for, and it is the slowest thing in a run - a browser launch, a wait
for the scene to settle, then the export dialog - hence the spinner.

Two kinds of board, because a link decides which editor opens. A share link opens the
editor itself, which has the export dialog. An Excalidraw+ read-only link opens a viewer
that has no menu at all - only a zoom - so there is no dialog to drive: that page ships
the whole scene in its own html, and the fix is to read it from there and hand it to
excalidraw.com, which does have the dialog. See :func:`_readonly_scene`.

SVG rather than PNG, because the agent reads a diagram far better that way: the export
keeps every label of the board as text and every box as a shape with coordinates, so the
model reads the names of the models and the arrows between them instead of inferring
them from pixels. It is the same dialog and the same cost, and an SVG stays legible at
any zoom for the human who opens the artifact afterwards.

Nothing here writes to stdout but the one question it has to ask. The browser is chatty
by nature and every step of it is a debug detail: on a normal run the user sees a
spinner, and with ``--log-level debug`` they see why the export took as long as it did,
or where it gave up. The exception is a board that will not open for a browser signed in
to nothing - a private Excalidraw+ one - which is asked for by hand while the developer
is still at the terminal. See :func:`_ask_for_exports`.

This lives in the base AI plugin, not in a command of its own: the ``excalidraw-export``
command is the agent's way in, but a run assembling a prompt reuses the same export and
the same url-finding, so both belong where every AI command can reach them.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import unicodedata
from contextlib import contextmanager
from html import unescape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from odev.common import progress
from odev.common.console import console
from odev.common.logging import logging


if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

EXCALIDRAW_URL = re.compile(
    r"https://(?:app\.|link\.|plus\.)?excalidraw\.com/[A-Za-z0-9/_\-#=?&]+(?:,[A-Za-z0-9_\-]+)?"
)
"""Matches an Excalidraw board linked from a task description.

Every host Excalidraw hands a board out on, because which one a link carries says
nothing about the board: ``excalidraw.com/#json=<id>,<key>`` is what the app itself
shares and what the Odoo editor stores for a Draw block, ``link.excalidraw.com/l/...``
is the shortened form of that same share, ``plus.excalidraw.com`` an Excalidraw+ board,
and ``app.`` an alias of the first. A link that is not the plain share redirects to one,
which the browser follows on its own - so all of them are worth opening.

The trailing group is the encryption key of a collaboration link, whose fragment reads
``#room=<id>,<key>``. Without it the board cannot be decrypted and the export opens an
empty canvas, so the comma is taken only when a key actually follows it - never the
comma that merely ends a sentence the url happens to sit in.
"""

DIAGRAM_FILENAME = "excalidraw-{index}-{name}{suffix}"
"""What an exported diagram is called on disk.

Numbered in the order the description links it, then named after the board itself -
Excalidraw suggests the title it was saved under, which is the only thing that tells
four diagrams of the same task apart. The prompt names the files it hands the agent, so
a board called "Dev flows" is a diagram the agent can refer to by name rather than by
the position it happened to hold in the description.

The suffix is the exporter's: SVG for a board exported here, and whatever the developer
handed over for one they exported themselves.
"""

FALLBACK_NAME = "diagram"
"""Stands in for the board's name when Excalidraw suggests none worth keeping."""

NAME_MAX_LENGTH = 60
"""How much of a board's name to keep, in characters, so a path stays a path."""

JOIN_TIMEOUT = 5000
"""How long to wait for the "Join room" dialog a shared board opens with, in ms."""

NETWORK_TIMEOUT = 15000
"""How long to wait for the images of a scene to load, in ms."""

REPAINT_DELAY = 500
"""How long to let the canvas repaint once its images have landed, in ms."""

CANVAS_TIMEOUT = 30000
"""How long to wait for a board to draw itself, in ms."""

POLL_DELAY = 250
"""How often to look at a page waiting to settle on one outcome or the other, in ms."""

INACCESSIBLE = re.compile(r"could ?n.t open this scene", re.IGNORECASE)
"""What the Excalidraw+ viewer puts on the page in place of a board it may not show.

Worth recognising rather than waiting out. That page is a client-side app: it fetches
the scene from Excalidraw's own backend, which answers 404 for a board that is not
public unless the request carries a session - and the browser started here is a fresh
profile that has none. The canvas then never appears, so without this the export sits
out its whole timeout only to end up no better informed than when it started.
"""

PRIVATE_BOARD = "the board is private and this browser is signed in to nothing, so it only opens for you"
"""Why such a board did not open, in the words the developer is asked to act on."""

NO_FILE_PICKER = "delete window.showSaveFilePicker; delete window.showOpenFilePicker;"
"""Hide the File System Access API from the page, so a save becomes a download.

Excalidraw saves through ``showSaveFilePicker`` wherever the browser has it - Chrome
does - and a headless one has no picker to show, so the call aborts and the export ends
in nothing at all: no file, no error on the page, just a download event that never
fires. Taking the api away leaves Excalidraw on its other branch, the anchor download
every browser without the api gets, which is the one Playwright can intercept.
"""

EDITOR_URL = "https://excalidraw.com"
"""Where a scene we hold ourselves is exported, having no board of its own to open."""

FLIGHT_CHUNK = re.compile(r'self\.__next_f\.push\(\[\d+,\s*("(?:[^"\\]|\\.)*")')
"""One chunk of the payload the read-only viewer, a Next.js app, renders itself from.

Read out of the html of the page rather than out of the array those calls fill, because
React has drained that array by the time the board is on screen. The chunks live on in
the script tags that pushed them, each a javascript string literal - which is json, so
``json.loads`` is what gives back the text one holds.
"""

SCENE_CONTENTS = '"sceneContents":'
"""Where, in that payload, the scene of the board itself starts.

The value is an ``.excalidraw`` document, the very thing the editor's own "Save to file"
writes: type, version, appState and every element. Which is why the export can go
through the editor unchanged - it is handed a scene it recognises as one of its own.
"""

SCENE_METADATA = '"sceneMetadata":'
"""Where, in that payload, what Excalidraw+ knows *about* the board starts.

Only for its ``name``: a read-only export has no filename to be suggested from, so
without it every diagram of a task would land as ``excalidraw-<n>-diagram.svg``.
"""


class ExportError(Exception):
    """Why a board could not be exported, in words worth putting in front of a person.

    Only raised for the failures that have an answer. A private board is one the
    developer at the terminal exports by hand in a couple of seconds, and saying so is
    worth more than the generic warning every other failure gets - a timeout out of
    Playwright names an element that did not appear, which tells them nothing they can
    act on.
    """


def find_diagram_urls(description: str | None) -> list[str]:
    """Return every Excalidraw board linked from ``description``, in order, once each.

    A task may link several - one per flow, or a board per iteration of the same one -
    and taking only the first quietly loses the rest.

    Read from the html rather than from the text it becomes: a board pasted in the Odoo
    editor may still be a ``data-embedded="draw"`` block holding its url in a json
    attribute, which the html-to-text pass drops entirely. Entities are resolved first,
    so a query string written ``&amp;`` in the markup is followed as the ``&`` it is.
    """
    if not description:
        return []

    return list(dict.fromkeys(EXCALIDRAW_URL.findall(unescape(description))))


def export_diagrams(urls: list[str], directory: Path, odev=None, interactive: bool = True) -> list[Path]:
    """Export each board of ``urls`` to an SVG in ``directory``, and return the paths.

    One browser for all of them: launching it is most of the cost, and a task linking
    four diagrams should not pay it four times. A board that cannot be exported is
    skipped with a warning rather than failing the run - an analysis without its diagram
    is worth more than no analysis - but not before the developer has been offered the
    chance to hand it over themselves, which :func:`_ask_for_exports` is for.

    :param odev: The odev instance, to reuse the Chrome it provisions. Falls back on
        Playwright's own bundled Chromium when not given, or when odev has none.
    :param interactive: Whether a board that did not export may be asked for by hand.
        True when a developer is at the terminal, as on an eager export before the agent
        runs. False when this is the agent itself calling in from inside the sandbox,
        where there is no one to prompt: a board that will not open is simply reported
        and left out, which the ``excalidraw-export`` command relies on.
    """
    if not urls:
        return []

    # Every board starts out unexported and leaves this list once it is on disk, so a
    # browser that never starts at all leaves them all to be asked for, the same way a
    # browser that started and could not open one does.
    missing: list[tuple[int, str]] = list(enumerate(urls, start=1))
    exported: dict[int, Path] = {}

    try:
        with _browser(odev) as browser:
            for index, url in list(missing):
                label = f"Exporting Excalidraw diagram {index}/{len(urls)}"

                try:
                    with progress.spinner(label):
                        diagram = _export_one(browser, url)
                except ExportError as e:
                    logger.warning(f"Could not export the Excalidraw diagram at {url}: {e}.")
                    continue

                if diagram is None:
                    logger.warning(f"Could not export the Excalidraw diagram at {url}.")
                    continue

                suggested, svg = diagram
                path = directory / DIAGRAM_FILENAME.format(index=index, name=_slugify(suggested), suffix=".svg")
                path.write_bytes(svg)
                exported[index] = path
                missing.remove((index, url))
                logger.debug(f"Exported {url} to {path}")
    except Exception as e:  # noqa: BLE001 - a missing diagram must not cost us the run
        logger.warning(f"Could not start a browser to export the Excalidraw diagrams: {e}")

    # Asked only once the browser and Playwright are both gone: InquirerPy drives its
    # prompts with asyncio.run(), which throws while Playwright's own loop is running.
    if interactive:
        exported.update(_ask_for_exports(missing, directory))

    if exported:
        logger.info(f"{len(exported)} Excalidraw diagram(s) ready for the agent.")

    return [exported[index] for index in sorted(exported)]


def _ask_for_exports(missing: list[tuple[int, str]], directory: Path) -> dict[int, Path]:
    """Ask for the boards that did not export, as files the developer exports by hand.

    Some failures no retry gets past. A private Excalidraw+ board opens for the people
    it was shared with and for nobody else, and the browser started here is a fresh
    profile signed in to nothing - while the developer who ran the command is signed in,
    and getting the picture out costs them one Ctrl+Shift+E.

    Asked here rather than reported afterwards because here the agent has not launched
    yet: the diagram is usually where the architecture of the task actually is, and an
    analysis written around the hole it leaves is the one thing the run was not for.
    Skipped when prompts are bypassed (``--headless``), there being nobody to ask.
    """
    if not missing or console.bypass_prompt:
        return {}

    logger.info(
        "A board that did not export can still be handed over: open it, export it with Ctrl+Shift+E as SVG, "
        "and give the path to the file when asked."
    )

    given: dict[int, Path] = {}

    for index, url in missing:
        if not console.confirm(f"Export {url} yourself and hand the file over?", default=True):
            continue

        source = _ask_for_file()

        if source is None:
            continue

        path = directory / DIAGRAM_FILENAME.format(
            index=index, name=_slugify(source.name), suffix=source.suffix.lower()
        )
        shutil.copyfile(source, path)
        given[index] = path
        logger.info(f"Took {source} as diagram {index}.")

    return given


def _ask_for_file() -> Path | None:
    """Ask for the file a board was exported to, until it names one. None to go on without.

    Asked again rather than given up on when the answer names nothing: the path is typed
    by hand from wherever the browser dropped the file, and a typo in it is not a change
    of mind about handing the diagram over.
    """
    while True:
        answer = console.filepath("Path to the exported diagram (empty to skip)")

        if not answer:
            return None

        source = Path(answer).expanduser()

        if source.is_file():
            return source

        logger.warning(f"{source} is not a file.")


def _slugify(filename: str) -> str:
    """Return the name Excalidraw saved a board under, as a name of our own.

    Accents are folded rather than dropped, so a board named in French keeps the word
    it was named with instead of the holes its accents leave behind.
    """
    stem = unicodedata.normalize("NFKD", Path(filename).stem).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:NAME_MAX_LENGTH].strip("-") or FALLBACK_NAME


@contextmanager
def _browser(odev=None) -> Iterator[Any]:
    """Start Playwright and yield a Chrome instance, closing both afterwards.

    Chrome, and where possible the very build odev already provisions for tours - the
    version Runbot pins - rather than a browser of our own. It is what Odoo is developed
    against, and reusing it means no second browser to install: `playwright install` is
    only needed when odev has no Chrome to lend.
    """
    # Imported here rather than at module level: Playwright is a heavy import, and a
    # task linking no diagram should not pay for it just to load the plugin.
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    executable = _odev_chrome(odev)

    logger.debug(f"Starting Playwright (Chrome{f' at {executable}' if executable else ', bundled'})")
    playwright = sync_playwright().start()

    try:
        browser = playwright.chromium.launch(headless=True, executable_path=executable)

        try:
            yield browser
        finally:
            browser.close()
    finally:
        playwright.stop()
        logger.debug("Stopped Playwright")


def _odev_chrome(odev=None) -> str | None:
    """Return the Chrome odev provisions, or None to use Playwright's own.

    Under a spinner because the first call downloads the browser, and ``Chrome.provision``
    captures the output of the npx doing it: without one, a first run sits silent for the
    length of a 150MB download with nothing on screen to say why. The message says which
    download it is, since a wait that explains itself is a different thing from a hang.
    """
    if odev is None:
        return None

    try:
        from odev.common.browsers import Chrome  # noqa: PLC0415

        with progress.spinner("Preparing the browser") as status:
            chrome = Chrome(odev)

            if not chrome.executable.exists():
                status.update(
                    f"Downloading Chrome {chrome.version} - first run only, "
                    "and shared with the browser odev runs tours with"
                )

            executable = chrome.provision()
    except Exception as e:  # noqa: BLE001 - Playwright's own Chromium is a fine fallback
        logger.debug(f"Could not provision odev's Chrome: {e}", exc_info=True)
        return None

    if not executable:
        logger.debug("odev has no Chrome to lend, falling back on Playwright's own Chromium.")

    return str(executable) if executable else None


def _export_one(browser: Any, url: str) -> tuple[str, bytes] | None:
    """Open ``url`` and export the board it holds, whichever kind of page it is.

    Returns the name the board is saved under, and the image bytes.

    :raises ExportError: when the page says why it will not open, so the caller can put
        that in front of the developer instead of only that something failed.
    """
    page = _new_page(browser)

    try:
        logger.debug(f"Loading {url}")
        page.goto(url)

        scene = _readonly_scene(page)

        if scene is not None:
            logger.debug("Read-only board: exporting its scene through the editor")
            return _export_scene(browser, scene)

        _join_room(page)

        logger.debug("Waiting for the drawing")
        _wait_for_canvas(page)
        _settle(page)

        return _download_svg(page)
    except ExportError:
        raise
    except Exception as e:  # noqa: BLE001 - reported by the caller, per diagram
        logger.debug(f"Export of {url} failed: {e}", exc_info=True)
        return None
    finally:
        page.close()


def _new_page(browser: Any, scene: dict[str, Any] | None = None) -> Any:
    """Open a page with the file picker hidden, and ``scene`` waiting for the editor.

    A scene is planted in local storage rather than opened as a file, because that is
    where the editor looks first: it restores the board it was last left on, before it
    has drawn anything, so the canvas comes up on our scene and no dialog is involved.
    Written in an init script so it is there before the editor's own scripts run.
    """
    page = browser.new_page()
    page.add_init_script(NO_FILE_PICKER)

    if scene is not None:
        elements = json.dumps(json.dumps(scene.get("elements", [])))
        app_state = json.dumps(json.dumps(scene.get("appState", {})))
        page.add_init_script(
            # Guarded: local storage throws on the about:blank a page starts its life on.
            f"try {{ localStorage.setItem('excalidraw', {elements});"
            f" localStorage.setItem('excalidraw-state', {app_state}); }} catch (e) {{}}"
        )

    return page


def _readonly_scene(page: Any) -> dict[str, Any] | None:
    """Return the scene an Excalidraw+ read-only viewer ships in its own page, if any.

    None for every other kind of board - the editor is on the page, and driving its
    dialog is both cheaper and truer to what the board looks like.

    The images of a scene do not come along: the viewer holds them encrypted and fetches
    the key on its own terms, so they export as the empty frames they are here. Every
    label, shape and arrow does, which is what the agent reads a diagram for.
    """
    try:
        payload = "".join(json.loads(chunk) for chunk in FLIGHT_CHUNK.findall(page.content()))
    except Exception as e:  # noqa: BLE001 - not a viewer page, then
        logger.debug(f"Could not read the payload of the page: {e}")
        return None

    scene = _json_at(payload, SCENE_CONTENTS)

    if not isinstance(scene, dict) or "elements" not in scene:
        return None

    metadata = _json_at(payload, SCENE_METADATA)
    name = metadata.get("name") if isinstance(metadata, dict) else None

    if name:
        # What the editor suggests as a filename, and so what the diagram is named after.
        scene.setdefault("appState", {})["name"] = name

    logger.debug(f"Read {len(scene['elements'])} elements of read-only board {name or '(unnamed)'}")

    return scene


def _json_at(payload: str | None, marker: str) -> Any:
    """Return the json object that follows ``marker`` in ``payload``, or None.

    The payload is a stream of react output rather than a document: the objects that
    interest us sit somewhere inside it, and only a decoder that stops of its own accord
    at the end of a value can pick one out of it.
    """
    if not payload or marker not in payload:
        return None

    try:
        return json.JSONDecoder().raw_decode(payload, payload.index(marker) + len(marker))[0]
    except ValueError as e:
        logger.debug(f"Could not read {marker} out of the page: {e}")
        return None


def _export_scene(browser: Any, scene: dict[str, Any]) -> tuple[str, bytes] | None:
    """Export a scene we hold ourselves, by handing it to the editor on excalidraw.com."""
    page = _new_page(browser, scene=scene)

    try:
        logger.debug(f"Loading the editor at {EDITOR_URL}")
        page.goto(EDITOR_URL)
        page.locator("canvas").first.wait_for(state="visible")
        _settle(page)

        return _download_svg(page)
    finally:
        page.close()


def _join_room(page: Any) -> None:
    """Get past the "Join room" dialog a shared board opens on, if there is one.

    The canvas stays behind it, and an export driven over it exports nothing.
    """
    try:
        join = page.get_by_role("button", name=re.compile("Join", re.IGNORECASE))
        join.wait_for(state="visible", timeout=JOIN_TIMEOUT)
        join.click()
        logger.debug("Joined the room")
        page.wait_for_timeout(2000)
    except Exception:  # noqa: BLE001 - most boards open without one
        logger.debug("No 'Join' dialog, proceeding")


def _wait_for_canvas(page: Any) -> None:
    """Wait until the board is on screen, or until the page admits it will not be.

    Both outcomes are the app's own and both arrive asynchronously, but only one of them
    is an element Playwright knows how to wait on - hence the poll. Waiting on the canvas
    alone turns a board that says plainly it cannot be opened into a timeout that says
    nothing, thirty seconds later.
    """
    canvas = page.locator("canvas").first
    refused = page.get_by_text(INACCESSIBLE).first
    deadline = time.monotonic() + CANVAS_TIMEOUT / 1000

    while time.monotonic() < deadline:
        if canvas.count() and canvas.is_visible():
            return

        if refused.count():
            raise ExportError(PRIVATE_BOARD)

        page.wait_for_timeout(POLL_DELAY)

    raise ExportError("the board did not draw itself in time")


def _settle(page: Any) -> None:
    """Wait for the scene to finish arriving on the canvas.

    The images of a scene are fetched asynchronously after the canvas becomes visible:
    exporting right away bakes their "broken image" placeholder into the export instead
    of the picture. Wait for the network to settle rather than guessing a delay; the
    short wait after is for the canvas to repaint once the data lands, which is neither
    a network nor a DOM event to wait on.
    """
    try:
        page.wait_for_load_state("networkidle", timeout=NETWORK_TIMEOUT)
    except Exception:  # noqa: BLE001
        logger.debug("Network did not settle in time, exporting anyway")

    page.wait_for_timeout(REPAINT_DELAY)


def _download_svg(page: Any) -> tuple[str, bytes]:
    """Drive the editor's own SVG export, and return the name it saves under and bytes."""
    logger.debug("Opening the export dialog")
    page.keyboard.press("Control+Shift+E")

    with page.expect_download() as download:
        page.locator('[aria-label="Export to SVG"]').click()

    return download.value.suggested_filename, Path(download.value.path()).read_bytes()
