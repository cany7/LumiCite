#TODO: finish data prep, core pipeline

.PHONY: help setup check-env download extract-images chunk embed index alt-text pipeline all clean clean-all

# Default target: show help
help:
	@echo "Project - Available targets:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Install Python dependencies from requirements.txt"
	@echo "  make check-env       - Verify required environment variables are set"
	@echo ""
	@echo "Data Preparation:"
	@echo "  make download       - Download PDFs from metadata.csv URLs to data/pdf/"
	@echo "  make extract-images - Extract figures/tables from PDFs to data/figures/"
	@echo ""
	@echo "Core Pipeline:"
	@echo "  make chunk          - Build chunks.json from PDFs using Docling"
	@echo "  make embed          - Generate embeddings.jsonl from chunks.json"
	@echo "  make index          - Build FAISS index from embeddings"
	@echo ""
	@echo "Alt-Text Generation:"
	@echo "  make alt-text       - Generate alt text for extracted figures (requires GEMINI_API_KEY)"
	@echo ""
	@echo "Convenience:"
	@echo "  make pipeline       - Run full text processing pipeline (chunk → embed → index)"
	@echo "  make all            - Run complete pipeline including images and alt-text"
	@echo "  make clean          - Remove generated artifacts (indexes, embeddings, alt-text)"
	@echo "  make clean-all      - Clean + remove extracted images and chunks"
	@echo ""

# Install Python dependencies
setup:
	@echo "Installing dependencies from requirements.txt..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Check required environment variables
check-env:
	@echo "Checking environment variables..."
	@if [ -z "$$GEMINI_API_KEY" ]; then \
		echo "⚠️  Warning: GEMINI_API_KEY is not set (required for alt-text generation)"; \
	else \
		echo "✅ GEMINI_API_KEY is set"; \
	fi

# Download PDFs from metadata.csv URLs
download:
	@echo "Downloading PDFs from metadata.csv..."
	python -c "from src.ingestion.load_files import load_data; load_data()"
	@echo "✅ PDF download complete"

# Extract figures/tables from PDFs
extract-images: check-pdfs
	@echo "Extracting figures and tables from PDFs..."
	python src/ingestion/image_extract.py
	@echo "✅ Image extraction complete"

# Check if PDFs exist before extracting images
check-pdfs:
	@if [ ! -d "data/pdf" ] || [ -z "$$(ls -A data/pdf/*.pdf 2>/dev/null)" ]; then \
		echo "❌ Error: No PDFs found in data/pdf/. Run 'make download' first."; \
		exit 1; \
	fi

# Build chunks.json from PDFs using Docling
chunk: check-pdfs
	@echo "Building chunks.json from PDFs..."
	python -m src.ingestion.JSON_builder
	@echo "✅ Chunking complete"

# Generate embeddings.jsonl from chunks.json
embed: check-chunks
	@echo "Generating embeddings from chunks.json..."
	python src/ingestion/embedder.py
	@echo "✅ Embedding generation complete"

# Check if chunks.json exists
check-chunks:
	@if [ ! -f "data/JSON/chunks.json" ]; then \
		echo "❌ Error: chunks.json not found. Run 'make chunk' first."; \
		exit 1; \
	fi

# Build FAISS index from embeddings
index: check-embeddings
	@echo "Building FAISS index from embeddings..."
	python src/indexing/vector_store.py
	@echo "✅ FAISS index built"

# Check if embeddings.jsonl exists
check-embeddings:
	@if [ ! -f "data/JSON/embeddings.jsonl" ]; then \
		echo "❌ Error: embeddings.jsonl not found. Run 'make embed' first."; \
		exit 1; \
	fi

# Generate alt text for extracted figures
alt-text: check-env check-figures
	@if [ -z "$$GEMINI_API_KEY" ]; then \
		echo "❌ Error: GEMINI_API_KEY environment variable is not set."; \
		echo "   Set it with: export GEMINI_API_KEY=your_key_here"; \
		exit 1; \
	fi
	@echo "Generating alt text for figures..."
	python src/ingestion/ocr.py
	@echo "✅ Alt-text generation complete"

# Check if figures export summary exists
check-figures:
	@if [ ! -f "data/figures/export_summary.csv" ]; then \
		echo "❌ Error: export_summary.csv not found. Run 'make extract-images' first."; \
		exit 1; \
	fi

# Run full text processing pipeline
pipeline: chunk embed index
	@echo "✅ Full text pipeline complete"

# Run complete pipeline including images and alt-text
all: download extract-images pipeline alt-text
	@echo "✅ Complete pipeline finished"

# Remove generated artifacts (indexes, embeddings, alt-text outputs)
clean:
	@echo "Cleaning generated artifacts..."
	@rm -f data/my_faiss.index
	@rm -f data/text_data.pkl
	@rm -f data/JSON/embeddings.jsonl
	@rm -f data/JSON/alt_text.json
	@rm -f data/figures/alt_text.csv
	@echo "✅ Clean complete"

# Clean + remove extracted images and chunks
clean-all: clean
	@echo "Cleaning extracted images and chunks..."
	@rm -f data/figures/*.png
	@rm -f data/JSON/chunks.json
	@echo "✅ Clean-all complete"
