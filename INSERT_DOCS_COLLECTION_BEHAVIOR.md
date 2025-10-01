# Analysis of insert_docs_qdrant.py Behavior with Collection "manuali"

## Issue Identified

The `insert_docs_qdrant.py` script **does** cancel or erase previous documents added to a collection under certain conditions:

1. When using `--embedding-method fastembed` with the default `--embedding-dim 1024`
2. If the collection already exists with different vector dimensions
3. The script detects a dimension mismatch and deletes/recreates the entire collection

## Root Cause

When running:
```bash
uv run insert_docs_qdrant.py https://gofastmcp.com/sitemap.xml --collection manuali --embedding-method fastembed
```

The script uses:
- FastEmbed model: `BAAI/bge-small-en` (384 dimensions)
- Default embedding dimension: 1024 (from `--embedding-dim` argument)

This creates a dimension mismatch, causing the script to delete and recreate the collection.

When running the second command:
```bash
uv run insert_docs_qdrant.py https://doc.agentscope.io --collection manuali --embedding-method fastembed
```

The same issue occurs, erasing any previously inserted documents.

## Solution Implemented

Modified `insert_docs_qdrant.py` to:

1. Automatically detect the correct dimensions for FastEmbed models
2. Use the actual model dimensions (384 for `BAAI/bge-small-en`) instead of the default 1024
3. Prevent unnecessary collection recreation when dimensions match

## Key Changes Made

1. Added `get_fastembed_dimensions()` function to retrieve actual model dimensions
2. Modified the main function to use correct dimensions for FastEmbed
3. Updated the embedding dimension parameter passed to `insert_documents_to_qdrant()`

## Recommendation

To prevent document loss when adding to existing collections:

1. Always specify the correct `--embedding-dim` when using FastEmbed:
   ```bash
   uv run insert_docs_qdrant.py https://gofastmcp.com/sitemap.xml --collection manuali --embedding-method fastembed --embedding-dim 384
   uv run insert_docs_qdrant.py https://doc.agentscope.io --collection manuali --embedding-method fastembed --embedding-dim 384
   ```

2. Or use the updated script which automatically handles dimension detection

This ensures that documents from both sources will be added to the same collection without erasing previous content.