#!/usr/bin/env python3
"""
Script to check embedding dimensions for FastEmbed models.
"""

try:
    from fastembed import TextEmbedding
    from fastembed.common.model_description import DenseModelDescription
    
    # Check dimensions for the default FastEmbed model
    model_name = "BAAI/bge-small-en"
    embedding_model = TextEmbedding(model_name)
    
    # Get model description
    model_description: DenseModelDescription = embedding_model._get_model_description(model_name)
    
    print(f"Model: {model_name}")
    print(f"Dimensions: {model_description.dim}")
    print(f"Description: {model_description.description}")
    
except ImportError:
    print("fastembed is not installed. Please install it with: pip install fastembed")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()