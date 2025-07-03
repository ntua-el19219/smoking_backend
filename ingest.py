import os
import argparse
import fitz
import requests
from bs4 import BeautifulSoup
import re

from chromaClient import get_chroma_collection

os.environ["ALLOW_CHROMA_TELEMETRY"] = "FALSE"

MIN_LENGTH = 200  # αγνοούμε πολύ μικρά κείμενα


def clean_text(text: str) -> str:
    # Αφαίρεση περιττών whitespaces, headers, artifacts
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def ingest_txt_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return clean_text(f.read())


def ingest_pdf_file(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    return clean_text(text)


def ingest_url(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Καθαρισμός από scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    return clean_text(soup.get_text(separator="\n"))


def ingest_to_knowledge_base(text: str, metadata: dict = None):
    if not text or len(text) < MIN_LENGTH:
        print(f"⚠️ Skipping short/empty text ({len(text)} chars)")
        return

    kb = get_chroma_collection("knowledge_base")
    kb.add_texts([text], metadatas=[metadata or {}])
    print(f"✅ Added to knowledge base ({len(text)} chars)")


def ingest_path(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        print(f"📄 Ingesting TXT: {path}")
        text = ingest_txt_file(path)
    elif ext == ".pdf":
        print(f"📕 Ingesting PDF: {path}")
        text = ingest_pdf_file(path)
    else:
        raise ValueError(f"❌ Unsupported file type: {ext}")

    ingest_to_knowledge_base(text, metadata={"source": path})


def ingest_from_url(url: str):
    print(f"🌐 Ingesting URL: {url}")
    text = ingest_url(url)
    ingest_to_knowledge_base(text, metadata={"source": url})


# -------------------------
# CLI ENTRYPOINT
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest files or URLs into the knowledge base (ChromaDB)")
    parser.add_argument("--file", type=str, help="Path to a .pdf or .txt file")
    parser.add_argument("--url", type=str, help="URL to ingest as knowledge")
    parser.add_argument("--folder", type=str, help="Folder path to ingest all .txt/.pdf files recursively") 

    args = parser.parse_args()

    if args.file:
        ingest_path(args.file)
    elif args.url:
        ingest_from_url(args.url)
    elif args.folder:
        for root, _, files in os.walk(args.folder):
            for file in files:
                if file.endswith((".txt", ".pdf")):
                    full_path = os.path.join(root, file)
                    try:
                        ingest_path(full_path)
                    except Exception as e:
                        print(f"⚠️ Error with {full_path}: {e}")
    else:
        print("❗ Please provide --file, --url or --folder")
        parser.print_help()
