---
name: download-deepwiki
description: Download a GitHub repository's DeepWiki documentation as a tar archive of markdown files. Use when the user asks for a DeepWiki export, wiki download, or offline documentation bundle for any public GitHub repo.
---

# Download DeepWiki

Export all pages from a GitHub repository's [DeepWiki](https://deepwiki.com) as a directory of markdown files, optionally packaged as a `.tar` archive.

## Prerequisites

- Python 3.11+ with `requests`, `beautifulsoup4`, and `html2text`
- The target repository must be indexed on DeepWiki (public repos are auto-indexed; private repos require Devin auth)

### Install dependencies (if not already available)

```bash
pip install requests beautifulsoup4 html2text
```

## Usage

The script lives at `.agents/skills/download-deepwiki/fetch_deepwiki.py` in this repo, but works for **any** GitHub repository.

### Download as tar archive

```bash
python3 .agents/skills/download-deepwiki/fetch_deepwiki.py <owner/repo> --tar
```

### Download to a custom directory

```bash
python3 .agents/skills/download-deepwiki/fetch_deepwiki.py <owner/repo> --output-dir ./my_wiki
```

### Examples

```bash
# This repo
python3 .agents/skills/download-deepwiki/fetch_deepwiki.py bckirkup/Crusher_to_the_Bridge --tar

# Any other public repo
python3 .agents/skills/download-deepwiki/fetch_deepwiki.py facebook/react --tar
python3 .agents/skills/download-deepwiki/fetch_deepwiki.py torvalds/linux --output-dir linux_wiki
```

## What It Does

1. **Discovers pages** — fetches the DeepWiki index page for the repo and extracts all wiki section slugs from the sidebar navigation.
2. **Downloads each page** — fetches the server-rendered HTML for every page with a polite 0.5s delay between requests.
3. **Converts to markdown** — extracts the main content (stripping nav/chrome) and converts to clean markdown with preserved links, tables, and code blocks.
4. **Packages output** — writes one `.md` file per wiki page; optionally creates a `.tar` archive.

## CLI Reference

```
usage: fetch_deepwiki.py [-h] [--output-dir DIR] [--tar] [--delay SECONDS] repo

positional arguments:
  repo               GitHub owner/repo (e.g. octocat/Hello-World)

options:
  --output-dir DIR   Output directory (default: <repo_name>_deepwiki/)
  --tar              Also produce a .tar archive
  --delay SECONDS    Pause between fetches (default: 0.5)
```

## Output Structure

```
Crusher_to_the_Bridge_deepwiki/
├── 1-architecture-overview.md
├── 1.1-infection-dynamics-bridge.md
├── 1.2-four-pathway-transmission-core.md
├── ...
└── 8-glossary.md
```

Each file is self-contained markdown with headings, tables, code blocks, and links back to source files on GitHub.

## Troubleshooting

- **"No wiki pages found"** — the repo may not be indexed on DeepWiki yet. Visit `https://deepwiki.com/<owner>/<repo>` in a browser to trigger indexing.
- **HTTP 429 (rate limit)** — increase `--delay` (e.g. `--delay 2`).
- **Private repo access** — DeepWiki requires Devin.ai OAuth for private repos; public repos work without auth.

## Sending to User

After generating the archive, attach it via `message_user`:

```python
# In Devin session
message_user(
    message="Here's the DeepWiki export.",
    attachments=["<repo_name>_deepwiki.tar"],
    block_on_user=True
)
```
