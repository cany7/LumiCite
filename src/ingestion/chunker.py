from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

def extract_pdf_chunks(pdf_path):
    # 1️⃣ Load and parse the PDF into a Docling document
    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    # Extract the actual document from the conversion result
    document = result.document

    # 2️⃣ Initialize the HybridChunker
    chunker = HybridChunker(
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",  # uses this model to estimate token length
        chunk_size=512,  # desired chunk size in tokens
        overlap=50,  # overlap between chunks
    )

    # 3️⃣ Chunk the document
    chunks = list(chunker.chunk(document))

    # 4️⃣ Inspect the output
    print(f"Generated {len(chunks)} chunks.\n")
    return chunks