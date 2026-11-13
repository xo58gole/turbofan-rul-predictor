"""Render docs/index.html and save the README screenshots to docs/img/.

Needs the extras:  pip install -r requirements-dev.txt  then  playwright install chromium
The Plotly CDN request is served from the copy bundled in the plotly package,
so this works offline.

Usage:  python scripts/screenshots.py [--full-dir some/folder]
"""

import argparse
from pathlib import Path

import plotly
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "docs" / "index.html").as_uri()
IMG = ROOT / "docs" / "img"
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"


def open_page(browser, width: int, scheme: str):
    page = browser.new_page(viewport={"width": width, "height": 900}, color_scheme=scheme, device_scale_factor=2)
    page.route("**/plotly*.min.js", lambda route: route.fulfill(path=str(PLOTLY_JS), content_type="text/javascript"))
    page.goto(PAGE)
    page.wait_for_function("document.querySelectorAll('.main-svg').length >= 5")
    page.wait_for_timeout(500)
    return page


def clip_between(page, first: str, last: str) -> dict:
    """A clip rectangle from the top of `first` to the bottom of `last`."""
    a = page.locator(first).first.bounding_box()
    b = page.locator(last).first.bounding_box()
    pad = 16
    return {"x": 0, "y": a["y"] - pad, "width": page.viewport_size["width"], "height": b["y"] + b["height"] - a["y"] + 2 * pad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-dir", type=Path, help="also save full-page shots here, for checking layout")
    args = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for scheme, name in (("light", "dashboard-hero.png"), ("dark", "dashboard-hero-dark.png")):
            page = open_page(browser, 1200, scheme)
            page.screenshot(path=str(IMG / name), clip=clip_between(page, "header", "section.card"), full_page=True)
            if scheme == "light":
                page.screenshot(path=str(IMG / "dashboard-charts.png"), clip=clip_between(page, ".grid2", ".grid2"), full_page=True)
            if args.full_dir:
                args.full_dir.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(args.full_dir / f"full-1200-{scheme}.png"), full_page=True)
            page.close()
        if args.full_dir:
            page = open_page(browser, 390, "light")
            page.screenshot(path=str(args.full_dir / "full-390-light.png"), full_page=True)
            page.close()
        browser.close()
    print(f"Saved screenshots to {IMG}")


if __name__ == "__main__":
    main()
