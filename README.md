# PDF-to-Markdown OCR Tool

This tool converts PDF files into Markdown by:

1. Converting each PDF page into an image
2. Using Google's Generative AI (Gemini) model to extract text and summarize non-textual elements as special tags

## Quick Start

Install uv and initialize your environment:

```bash
uv sync
```

This command installs all dependencies (including pdf2image, Pillow, google-generativeai, and dev tools like ruff, pre-commit, pytest) as specified in pyproject.toml.

Run the tool:

```bash
uv run python pdf_to_markdown.py input.pdf output.md --api_key=YOUR_API_KEY
```
