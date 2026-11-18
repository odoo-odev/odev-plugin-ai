"""Read the prompt from the clipboard instead of typing it on the command line.

``odev ai --paste`` hands the agent whatever was copied - typically a task description
pasted straight out of the browser. The HTML flavour of the clipboard is preferred over
plain text: it is what keeps the ``<img>`` links of a description, which a plain-text
copy has already lost. The markup is turned into text, with the image urls kept inline as
links for the agent to load on its own. Inline ``data:`` screenshots - a bug report pasted
as a base64 payload rather than an uploaded url - cannot ride on argv the way a url can
(one payload overruns the length a single command-line argument is allowed), so each is
written out to a temp file and linked by path, for the agent to open on its own.
"""

from __future__ import annotations

import base64
import binascii
import mimetypes
import os
import re
import subprocess
import tempfile
import urllib.parse

import html2text

from odev.common.logging import logging


logger = logging.getLogger(__name__)

HTML_TAG = re.compile(r"<[a-zA-Z/][^>]*>")
"""Enough to tell a clipboard that is markup from one that is plain text."""

# An ``<img>`` whose src is a ``data:`` payload rather than a url, with the payload captured
# as ``uri``. A url image is left as it stands; only these are rewritten, because one base64
# screenshot blows past the ~128kB the kernel allows a single command-line argument and the
# prompt rides on argv - so the payload is written to a file and the src swapped for its path.
DATA_URI_IMG = re.compile(
    r"""<img\b[^>]*\bsrc\s*=\s*["'](?P<uri>data:[^"']*)["'][^>]*>""", re.IGNORECASE
)

# A readable extension for the common image types a browser pastes, since
# ``mimetypes.guess_extension`` is platform-dependent (``.jpe`` for jpeg on some boxes).
_IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
}


class PasteError(Exception):
    """The clipboard could not be read, or held nothing usable."""


def _decode_data_uri(uri: str) -> tuple[bytes, str] | None:
    """Return the ``(bytes, mimetype)`` of a ``data:`` URI, or ``None`` if unreadable.

    The shape is ``data:<mimetype>[;base64],<payload>``. A base64 payload is decoded;
    the plain (percent-encoded) form is handled too, though a pasted screenshot is always
    base64. A payload that will not decode hands back ``None`` so the caller can drop it
    rather than let one malformed tag break the whole paste.
    """
    try:
        header, data = uri[len("data:") :].split(",", 1)
    except ValueError:
        return None

    is_base64 = header.endswith(";base64")
    mimetype = (header[: -len(";base64")] if is_base64 else header).split(";", 1)[0].strip().lower()
    mimetype = mimetype or "application/octet-stream"

    try:
        raw = base64.b64decode(data) if is_base64 else urllib.parse.unquote_to_bytes(data)
    except (binascii.Error, ValueError):
        return None

    return raw, mimetype


def _extension_for(mimetype: str) -> str:
    """The file extension to give a payload of ``mimetype`` (``.png``, ``.jpg``, ...)."""
    return _IMAGE_EXTENSIONS.get(mimetype) or mimetypes.guess_extension(mimetype) or ".img"


def _save_data_uri_images(content: str) -> str:
    """Write inline ``data:`` images out to files, leaving url images untouched.

    Content copied from the browser can carry a screenshot inline as a base64 ``data:``
    URI - no link to hand the agent, and a payload that alone can overrun the command line
    the prompt reaches the agent on. Each is decoded to a temp file and its ``src`` swapped
    for that path, so the ``<img>`` survives into the prompt as a link the agent can open
    (a pasted bug-report screenshot is often the most useful part). A payload that will not
    decode is dropped, as there is nothing to save or link. Url ``<img>`` tags are left be.

    The files are written to a temp directory that outlives this call - the agent reads
    them during its run - and is left for the OS to reclaim with the rest of its temp.
    """
    saved = 0
    directory: str | None = None  # created lazily, only once an image is actually found

    def replace(match: re.Match) -> str:
        nonlocal saved, directory

        decoded = _decode_data_uri(match.group("uri"))
        if decoded is None:
            return ""

        raw, mimetype = decoded
        if directory is None:
            directory = tempfile.mkdtemp(prefix="odev-ai-paste-")

        saved += 1
        path = os.path.join(directory, f"image-{saved}{_extension_for(mimetype)}")
        with open(path, "wb") as file:
            file.write(raw)

        # Swap the base64 src for the file path; html2text then renders ``![alt](path)``.
        return match.group(0).replace(match.group("uri"), path, 1)

    content = DATA_URI_IMG.sub(replace, content)

    if saved:
        logger.info(f"Saved {saved} inline image(s) from the pasted prompt to {directory}.")

    return content


def _read_clipboard() -> str:
    """Return what the clipboard holds, read through whatever tool is on the box.

    The HTML flavour is asked for first, across ``wl-paste`` (Wayland) then ``xclip``
    (X11) - it is what keeps the image links a plain-text paste drops - and plain text is
    taken only when no tool returns HTML, so a description with no markup still goes
    through. A missing tool or an empty answer just hands over to the next. Nothing
    readable at all is an error pointing at the fix, the clipboard being a graphical thing
    a bare SSH session does not have.
    """
    for command in (
        ["wl-paste", "--type", "text/html"],
        ["xclip", "-selection", "clipboard", "-o", "-t", "text/html"],
        ["wl-paste"],
        ["xclip", "-selection", "clipboard", "-o"],
    ):
        try:
            result = subprocess.run(command, capture_output=True, check=True)
        except FileNotFoundError:
            # The tool is not installed: try the next one.
            continue
        except subprocess.CalledProcessError:
            # The tool ran but gave nothing back - that flavour is not on the clipboard,
            # or no display it can reach (wl-paste on an X11 session): try the next one,
            # which may be the one that fits this box.
            continue

        if result.stdout.strip():
            return result.stdout.decode("utf-8", errors="replace")

    raise PasteError(
        "Could not read the clipboard. Copy the text first, and make sure wl-clipboard (Wayland) "
        "or xclip (X11) is installed; over SSH there is no clipboard."
    )


def _to_text(content: str) -> str:
    """Return the clipboard content as text, images kept as links.

    Plain text is returned as it stands: run through html2text it would come back with
    its ``<``, ``&`` and ``*`` read as markup. HTML has its inline ``data:`` images
    written out to files first, then is turned into markdown text with every ``<img>``
    link kept - url and saved-file alike - the agent loads them on its own.
    """
    if not HTML_TAG.search(content):
        return content.strip()

    content = _save_data_uri_images(content)

    h = html2text.HTML2Text()
    h.ignore_links = True
    # Keep the images as links: the url images stay in the text as ``![alt](url)`` for
    # the agent to load by itself.
    h.ignore_images = False
    # No line wrapping: html2text wraps at 78 columns by default, which would fold a long
    # url or saved-file path mid-link and break the path the agent is meant to open.
    h.body_width = 0
    return h.handle(content).strip()


def read_pasted_prompt() -> str:
    """Return the clipboard, parsed into a prompt the agent can read.

    The HTML flavour is preferred so image links survive; the markup is turned into text
    with its url images kept inline as links, for the agent to load by itself. Raises
    :class:`PasteError` when the clipboard cannot be read or holds nothing.
    """
    text = _to_text(_read_clipboard())

    if not text.strip():
        raise PasteError("The clipboard holds nothing to send.")

    return text
