"""Render math with typst.

# See

https://forum.typst.app/t/guide-render-typst-math-in-mkdocs/4326

## Usage

1. Install the markdown extensions pymdownx.arithmatex.
2. Add `math: typst` to pages' metadata.

## Requirements

- typst

"""

from __future__ import annotations

import html
import re
from functools import cache
from subprocess import CalledProcessError, run
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page


def should_render(page: Page) -> bool:
    return page.meta.get("math") == "typst"


def on_page_markdown(
    markdown: str, page: Page, config: MkDocsConfig, files: Files
) -> str | None:
    if should_render(page):
        assert "pymdownx.arithmatex" in config.markdown_extensions, (
            "Missing pymdownx.arithmatex in config.markdown_extensions. "
            "Setting `math: typst` requires it to parse markdown."
        )


def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str | None:
    if should_render(page):
        output = re.sub(
            r'<span class="arithmatex">(.+?)</span>', render_inline_math, output
        )

        output = re.sub(
            r'<div class="arithmatex">(.+?)</div>',
            render_block_math,
            output,
            flags=re.MULTILINE | re.DOTALL,
        )
        return output


def render_inline_math(match: re.Match[str]) -> str:
    src = html.unescape(match.group(1)).removeprefix(
        R"\(").removesuffix(R"\)").strip()
    typ = f"${src}$"
    return (
        '<span class="typst-math">'
        + fix_svg(typst_compile(typ))
        + for_screen_reader(typ)
        + "</span>"
    )


def render_block_math(match: re.Match[str]) -> str:
    src = html.unescape(match.group(1)).removeprefix(
        R"\[").removesuffix(R"\]").strip()
    typ = f"$ {src} $"
    return (
        '<div class="typst-math">'
        + fix_svg(typst_compile(typ))
        + for_screen_reader(typ)
        + "</div>"
    )


def for_screen_reader(typ: str) -> str:
    return f'<span class="sr-only">{html.escape(typ)}</span>'


def fix_svg(svg: bytes) -> str:
    """Fix the compiled SVG to be embedded in HTML

    - Strip trailing spaces
    - Support dark theme
    """
    return re.sub(
        r' (fill|stroke)="#000000"',
        r' \1="var(--md-typeset-color)"',
        svg.decode().strip(),
    )


@cache
def typst_compile(
    typ: str,
    *,
    prelude="#set page(width: auto, height: auto, margin: 0pt, fill: none)\n",
    format="svg",
) -> bytes:
    """Compile a Typst document

    https://github.com/marimo-team/marimo/discussions/2441
    """
    try:
        return run(
            ["typst", "compile", "-", "-", "--format", format],
            input=(prelude + typ).encode(),
            check=True,
            capture_output=True,
        ).stdout
    except CalledProcessError as err:
        raise RuntimeError(
            f"""
Failed to render a typst math:

```typst
{typ}
```

{err.stderr.decode()}
""".strip()
        )
