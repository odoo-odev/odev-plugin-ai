"""Read the prompt from the clipboard instead of typing it on the command line.

``odev ai --paste`` hands the agent whatever was copied - typically a task description
pasted straight out of the browser. The HTML flavour of the clipboard is preferred over
plain text: it is what keeps the ``<img>`` links of a description, which a plain-text
copy has already lost. The markup is turned into text, inline ``data:`` screenshots are
dropped (they are no link to fetch and a single one overruns the command line), and the
image urls are kept inline so the agent can load them itself.
"""

from __future__ import annotations

import re
import subprocess

import html2text

from odev.common.logging import logging


logger = logging.getLogger(__name__)

HTML_TAG = re.compile(r"<[a-zA-Z/][^>]*>")
"""Enough to tell a clipboard that is markup from one that is plain text."""

# An ``<img>`` whose src is a base64 ``data:`` payload rather than a url. A pasted prompt
# keeps its images as links for the agent to fetch; an inline data URI is no link, and a
# single screenshot of one blows past the 128kB the kernel allows a command line - so
# those are dropped rather than carried into the prompt.
DATA_URI_IMG = re.compile(r"""<img\b[^>]*\bsrc\s*=\s*["']data:[^"']*["'][^>]*>""", re.IGNORECASE)

# The image links left in a parsed (markdown) prompt, as ``![alt](url)``.
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*(\S+?)\s*\)")


class PasteError(Exception):
    """The clipboard could not be read, or held nothing usable."""


def _drop_data_uri_images(content: str) -> str:
    """Strip inline ``data:`` images from pasted HTML, leaving url images untouched.

    Content copied from the browser can carry a screenshot inline as a base64 ``data:``
    URI. That is no link to hand the agent, and the payload alone can overrun the command
    line the prompt reaches the agent on, so the tag is removed. An ``<img>`` pointing at
    a url is left in place, kept in the prompt for the agent to load.
    """
    content, dropped = DATA_URI_IMG.subn("", content)

    if dropped:
        logger.info(f"Dropped {dropped} inline data-URI image(s) from the pasted prompt (no link to fetch).")

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
    dropped first, then is turned into markdown text with the remaining ``<img>`` links
    kept - there is nothing fetched behind a paste, so the agent loads them itself.
    """
    if not HTML_TAG.search(content):
        return content.strip()

    content = _drop_data_uri_images(content)

    h = html2text.HTML2Text()
    h.ignore_links = True
    # Keep the images as links: a paste has no task behind it to fetch them from, so the
    # urls stay in the text for the agent to load.
    h.ignore_images = False
    return h.handle(content).strip()


def _image_links(text: str) -> list[str]:
    """The image urls kept inline in the parsed prompt, for the agent to load."""
    return MARKDOWN_IMAGE.findall(text)


def read_pasted_prompt() -> str:
    """Return the clipboard, parsed into a prompt the agent can read.

    The HTML flavour is preferred so image links survive; the text is returned with its
    url images kept inline, and a closing note tells the agent to load each one (a pasted
    prompt has nothing fetched behind it). Raises :class:`PasteError` when the clipboard
    cannot be read or holds nothing.
    """
    text = _to_text(_read_clipboard())

    if not text.strip():
        raise PasteError("The clipboard holds nothing to send.")

    if links := _image_links(text):
        joined = ", ".join(f"`{link}`" for link in links)
        text += (
            f"\n\nThe text above keeps {len(links)} image(s) as link(s): {joined}. "
            "Load each one and read it as the picture the text points to there, where the link stands."
        )

    return text
