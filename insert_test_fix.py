#!/usr/bin/env python3
"""
Script to insert test documents into Qdrant with the correct format to test our fix.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from openai import OpenAI

def get_qdrant_client():
    """Initialize and return a Qdrant client."""
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file")
    
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_embedding_client():
    """Initialize and return an OpenAI-compatible embedding client."""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )

def generate_embeddings(client, texts, model_name, dimensions):
    """Generate embeddings for a list of texts."""
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(
            model=model_name,
            input=batch,
            dimensions=dimensions
        )
        all_embeddings.extend([data.embedding for data in resp.data])
    return all_embeddings

def insert_test_documents():
    """Insert test documents into Qdrant with the correct format."""
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    qdrant_client = get_qdrant_client()
    embedding_client = get_embedding_client()
    
    # Test documents with content about the AddAndManageSingleTextDrawingForStudy function
    documents = [
        """### sc.AddAndManageSingleTextDrawingForStudy()

The sc.AddAndManageSingleTextDrawingForStudy() function adds or manages a single text drawing for a study. This function is used to display text on a chart that is associated with a specific study instance.

This function is particularly useful for displaying study-specific information such as parameter values, calculated values, or status messages directly on the chart where the study is applied.

Parameters:
- DrawingText: A string that specifies the text to be displayed.
- LineNumber: An integer that specifies the line number for the text drawing.
- HorizontalPosition: An integer that specifies the horizontal position of the text.
- VerticalPosition: A float that specifies the vertical position of the text.
- Color: An integer that specifies the color of the text.
- FontSize: An integer that specifies the font size of the text.
- FontBold: A boolean that specifies whether the text should be bold.

Returns:
- An integer handle to the text drawing, which can be used to modify or delete the drawing later.""",
        """### sc.AddAndManageSingleTextUserDrawnDrawingForStudy()

The sc.AddAndManageSingleTextUserDrawnDrawingForStudy() function is similar to sc.AddAndManageSingleTextDrawingForStudy() but is used for user-drawn text drawings. This function allows the user to interactively place text on the chart, and the text drawing is associated with the study instance.

This function is useful for allowing users to add annotations or notes to a chart that are specific to a study instance.

Parameters:
- DrawingText: A string that specifies the text to be displayed.
- LineNumber: An integer that specifies the line number for the text drawing.
- HorizontalPosition: An integer that specifies the horizontal position of the text.
- VerticalPosition: A float that specifies the vertical position of the text.
- Color: An integer that specifies the color of the text.
- FontSize: An integer that specifies the font size of the text.
- FontBold: A boolean that specifies whether the text should be bold.

Returns:
- An integer handle to the text drawing, which can be used to modify or delete the drawing later.""",
        """### Additional Information about Text Drawings in Sierra Chart

Text drawings in Sierra Chart are used to display textual information on charts. They can be static or dynamic, and can be associated with specific studies or chart elements. The AddAndManageSingleTextDrawingForStudy function is part of a family of functions that allow developers to programmatically create and manage text drawings.

Text drawings can be used for:
- Displaying parameter values
- Showing calculated results
- Providing status updates
- Adding annotations
- Creating interactive elements""",
    ]
    
    # Metadata for each document
    metadatas = [
        {
            "headers": "### sc.AddAndManageSingleTextDrawingForStudy()",
            "char_count": len(documents[0]),
            "word_count": len(documents[0].split()),
            "chunk_index": 0,
            "source": "https://www.sierrachart.com/index.php?page=doc/ACSIL_Members_Functions.html",
            "document": documents[0]  # This is the key fix - include the actual document content
        },
        {
            "headers": "### sc.AddAndManageSingleTextUserDrawnDrawingForStudy()",
            "char_count": len(documents[1]),
            "word_count": len(documents[1].split()),
            "chunk_index": 1,
            "source": "https://www.sierrachart.com/index.php?page=doc/ACSIL_Members_Functions.html",
            "document": documents[1]  # Include the actual document content
        },
        {
            "headers": "### Additional Information about Text Drawings in Sierra Chart",
            "char_count": len(documents[2]),
            "word_count": len(documents[2].split()),
            "chunk_index": 2,
            "source": "https://www.sierrachart.com/index.php?page=doc/ACSIL_Members_Functions.html",
            "document": documents[2]  # Include the actual document content
        }
    ]
    
    collection_name = "docs_test"
    model_name = "text-embedding-v3"
    embedding_dim = 1024
    
    print(f"Inserting {len(documents)} test documents into Qdrant collection '{collection_name}'...")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(embedding_client, documents, model_name, embedding_dim)
    print(f"Generated {len(embeddings)} embeddings with {embedding_dim} dimensions")
    
    # Delete collection if it exists
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection '{collection_name}'.")
    except:
        print(f"Collection '{collection_name}' does not exist.")
    
    # Create collection
    print(f"Creating collection '{collection_name}'...")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_dim, distance=models.Distance.COSINE)
    )
    
    # Upload points with pre-computed embeddings
    points = [
        models.PointStruct(
            id=i,
            vector=embeddings[i],
            payload=metadatas[i]  # This includes the document content
        )
        for i in range(len(documents))
    ]
    
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Successfully added {len(documents)} documents to Qdrant collection '{collection_name}'.")

if __name__ == "__main__":
    try:
        insert_test_documents()
        print("\nTest documents inserted successfully!")
    except Exception as e:
        print(f"\nError inserting test documents: {e}")
        exit(1)