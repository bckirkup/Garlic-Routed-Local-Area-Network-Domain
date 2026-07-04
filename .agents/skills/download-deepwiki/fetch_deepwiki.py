#!/usr/bin/env python3
"""Download all DeepWiki pages for a GitHub repository as markdown files.

Usage:
    python3 fetch_deepwiki.py <owner/repo> [--output-dir DIR] [--tar]

Examples:
    python3 fetch_deepwiki.py bckirkup/Crusher_to_the_Bridge --tar
    python3 fetch_deepwiki.py octocat/Hello-World --output-dir ./wiki_export
"""

import argparse
import json
import os
import re
import sys
import tarfile
import time

import requests
from bs4 import BeautifulSoup
import html2text


DEEPWIKI_BASE = "https://deepwiki.com"


def discover_pages(repo: str) -> list[str]:
    """Fetch the main wiki page and extract all page slugs from the sidebar."""
    url = f"{DEEPWIKI_BASE}/{repo}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    prefix = f"/{repo}/"
    slugs = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith(prefix):
            slug = href[len(prefix):]
            if slug and slug not in slugs:
                slugs.append(slug)
    return slugs


def fetch_page_html(repo: str, page_slug: str) -> str:
    """Fetch a single DeepWiki page and return raw HTML."""
    url = f"{DEEPWIKI_BASE}/{repo}/{page_slug}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def html_to_markdown(html_text: str) -> str:
    """Extract the main wiki content from DeepWiki HTML and convert to markdown."""
    soup = BeautifulSoup(html_text, "html.parser")

    # DeepWiki uses Next.js; try the __NEXT_DATA__ payload first
    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data:
        try:
            data = json.loads(next_data.string)
            props = data.get("props", {}).get("pageProps", {})
            for key in ("content", "markdown", "body", "wiki"):
                val = props.get(key)
                if isinstance(val, str) and len(val) > 100:
                    return val
                if isinstance(val, dict) and "content" in val:
                    return val["content"]
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: find the largest content div with headings
    main_content = None
    for selector in ["article", "main", ".prose", ".markdown-body"]:
        main_content = soup.select_one(selector)
        if main_content:
            break

    if not main_content:
        divs_with_headings = []
        for div in soup.find_all("div"):
            h_tags = div.find_all(re.compile(r"^h[1-6]$"))
            if h_tags:
                divs_with_headings.append((len(div.get_text()), div))
        if divs_with_headings:
            divs_with_headings.sort(key=lambda x: x[0], reverse=True)
            main_content = divs_with_headings[0][1]

    if not main_content:
        main_content = soup.body

    if main_content:
        for tag in main_content.find_all(["nav", "aside", "footer", "header", "script", "style"]):
            tag.decompose()
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = False
        converter.body_width = 0
        return converter.handle(str(main_content))

    return "(Content could not be extracted)"


def sanitize_filename(slug: str) -> str:
    """Convert a page slug to a safe filename."""
    return slug.replace("(", "").replace(")", "").replace("/", "_") + ".md"


def main():
    parser = argparse.ArgumentParser(
        description="Download a GitHub repo's DeepWiki as markdown files."
    )
    parser.add_argument("repo", help="GitHub owner/repo (e.g. octocat/Hello-World)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write markdown files (default: ./<repo_name>_deepwiki/)",
    )
    parser.add_argument(
        "--tar",
        action="store_true",
        help="Also create a .tar archive of the output directory",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between page fetches (default: 0.5)",
    )
    args = parser.parse_args()

    repo = args.repo.strip("/")
    repo_name = repo.split("/")[-1]

    output_dir = args.output_dir or f"{repo_name}_deepwiki"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Discovering pages for {repo}...")
    pages = discover_pages(repo)

    if not pages:
        print("ERROR: No wiki pages found. Is the repo indexed on DeepWiki?", file=sys.stderr)
        print(f"  Try visiting: {DEEPWIKI_BASE}/{repo}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pages)} pages. Downloading...")

    for i, slug in enumerate(pages):
        print(f"  [{i + 1}/{len(pages)}] {slug}")
        try:
            html = fetch_page_html(repo, slug)
            md = html_to_markdown(html)
            filepath = os.path.join(output_dir, sanitize_filename(slug))
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md)
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)
            filepath = os.path.join(output_dir, sanitize_filename(slug))
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {slug}\n\nError fetching page: {e}\n")

        if i < len(pages) - 1:
            time.sleep(args.delay)

    # Create tar if requested
    if args.tar:
        tar_path = f"{repo_name}_deepwiki.tar"
        print(f"\nCreating archive: {tar_path}")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(output_dir, arcname=f"{repo_name}_deepwiki")
        size_kb = os.path.getsize(tar_path) / 1024
        print(f"Archive created: {tar_path} ({size_kb:.1f} KB)")

    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
    )
    print(f"\nDone! {len(pages)} pages → {output_dir}/ ({total_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
