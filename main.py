#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
import shutil
import logging
import asyncio
from pdf2image import convert_from_path
from PIL import Image

from google import genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def convert_pdf_to_images(pdf_path, dpi=200):
    """
    Synchronously convert the given PDF file into a list of image file paths, one per page.
    """
    logging.info(f"Converting PDF '{pdf_path}' to images at {dpi} DPI...")
    tmpdir = tempfile.mkdtemp(prefix="pdf_to_images_")
    images = convert_from_path(pdf_path, dpi=dpi)

    image_paths = []
    for i, img in enumerate(images):
        page_num = i + 1
        img_path = os.path.join(tmpdir, f"page_{page_num}.png")
        img.save(img_path, "PNG")
        image_paths.append(img_path)

    logging.info(f"Converted PDF into {len(image_paths)} images.")
    return tmpdir, image_paths


async def ocr_single_image(client, image_path, page_num, model_name):
    """
    OCR a single image using the provided GenAI client (synchronously) in a thread executor.
    """
    image = Image.open(image_path)
    image.thumbnail((512, 512))

    # Updated prompt as requested:
    prompt = (
        "You are a document transcription assistant. "
        "I will provide you with an image of a page from a PDF. "
        "Your task is to extract all readable text from the image verbatim and present it in Markdown format. "
        "For any non-textual elements (such as diagrams, images, charts, or figures), do not transcribe their content as text. "
        "Instead, replace each non-textual element with a special tag like: `<NON_TEXT>Description_of_element</NON_TEXT>`. "
        "If something is unreadable, skip it. "
        f"This is page {page_num} of the document.\n\n"
    )

    def generate_content():
        response = client.models.generate_content(
            model=model_name, contents=[image, prompt]
        )
        return response.text.strip()

    result = await asyncio.to_thread(generate_content)
    return result


async def ocr_images_with_llm(image_paths, model_name, api_key, concurrency=5):
    """
    Asynchronously OCR all images using the LLM, with concurrency.
    """
    logging.info(f"Initializing GenAI client with model '{model_name}'...")
    client = genai.Client(api_key=api_key)

    sem = asyncio.Semaphore(concurrency)

    async def ocr_task(idx, image_path):
        async with sem:
            return await ocr_single_image(client, image_path, idx + 1, model_name)

    tasks = []
    for idx, image_path in enumerate(image_paths):
        tasks.append(ocr_task(idx, image_path))

    logging.info("Starting asynchronous OCR on images...")
    results = await asyncio.gather(*tasks)

    return "\n\n".join(results)


def write_markdown(output_md, text):
    """
    Synchronously write the given text to the specified Markdown file.
    """
    logging.info(f"Writing output to '{output_md}'...")
    try:
        with open(output_md, "w", encoding="utf-8") as md_file:
            md_file.write(text)
    except Exception as e:
        logging.error(f"Error writing Markdown file {output_md}: {e}")
        sys.exit(1)
    logging.info("Output written successfully.")


async def main_async(args):
    if not args.api_key:
        args.api_key = os.environ.get("GENAI_API_KEY", "")
        if not args.api_key:
            logging.error(
                "No API key provided. Please use --api_key or set GENAI_API_KEY env variable."
            )
            sys.exit(1)

    if not os.path.isfile(args.input_pdf):
        logging.error(f"Input PDF file not found: {args.input_pdf}")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_md)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Could not create output directory {output_dir}: {e}")
            sys.exit(1)

    tmpdir, image_paths = convert_pdf_to_images(args.input_pdf, dpi=args.dpi)

    try:
        ocr_text = await ocr_images_with_llm(
            image_paths, model_name=args.model_name, api_key=args.api_key, concurrency=5
        )
        write_markdown(args.output_md, ocr_text)
        logging.info(
            f"Successfully converted {args.input_pdf} to {args.output_md} using the LLM OCR approach."
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF file to Markdown using LLM OCR (Google GenAI) asynchronously."
    )
    parser.add_argument("input_pdf", help="Path to the input PDF file.")
    parser.add_argument("output_md", help="Path to the output Markdown file.")
    parser.add_argument(
        "--model_name",
        default="gemini-2.0-flash-exp",
        help="Name of the Gemini model to use.",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="DPI for converting PDF pages to images."
    )
    parser.add_argument("--api_key", default="", help="Your Google GenAI API key.")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
