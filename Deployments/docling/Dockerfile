FROM ghcr.io/docling-project/docling-serve-cu126:latest

# Set artifacts path
ENV DOCLING_ARTIFACTS_PATH=/artifacts
RUN mkdir -p /artifacts

# Download Granite model (no --cache-dir)
RUN docling download vlm granite_docling

