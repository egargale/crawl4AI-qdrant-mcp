from dotenv import load_dotenv
import os
import re

# Load environment variables from .env file
load_dotenv()

# Get Qdrant configuration from environment variables
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file")

from qdrant_client import QdrantClient, models
from openai import OpenAI

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

print("Successfully connected to Qdrant!")
print(f"Qdrant URL: {QDRANT_URL}")

# Initialize OpenAI client for Dashscope embeddings
embedding_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_URL"),
)

collection_name = "demo_collection"
# Using a model with 1024 dimensions
model_name = "text-embedding-v3"
embedding_dim = 1024

# Read the README.md file
try:
    with open("README.md", "r", encoding="utf-8") as f:
        readme_content = f.read()
except FileNotFoundError:
    raise FileNotFoundError("README.md file not found in the current directory")

def semantic_chunking(text, max_chunk_size=800, min_chunk_size=300):
    """
    Implement state-of-the-art chunking for semantic search.
    This approach respects document structure and semantic boundaries.
    """
    # Split text into sections based on markdown headers
    sections = re.split(r'(\n##+ .*\n)', text)
    
    chunks = []
    current_chunk = ""
    
    for section in sections:
        # If this is a header, start a new chunk
        if re.match(r'\n##+ .*\n', section):
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = section
        # If adding this section would exceed max_chunk_size, create a new chunk
        elif len(current_chunk) + len(section) > max_chunk_size and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = section
        else:
            current_chunk += section
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Further split any large chunks that exceed max_chunk_size
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split by paragraphs
            paragraphs = re.split(r'\n\s*\n', chunk)
            sub_chunk = ""
            for paragraph in paragraphs:
                if len(sub_chunk) + len(paragraph) > max_chunk_size and len(sub_chunk) >= min_chunk_size:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = paragraph
                else:
                    sub_chunk += "\n\n" + paragraph if sub_chunk else paragraph
            if sub_chunk:
                final_chunks.append(sub_chunk.strip())
    
    return final_chunks

# Apply semantic chunking
chunks = semantic_chunking(readme_content)
print(f"Created {len(chunks)} semantically meaningful chunks")

# Create payloads with chunks
payload = [{"document": chunk, "source": "README.md", "chunk_id": i} for i, chunk in enumerate(chunks)]
ids = list(range(len(chunks)))

# Generate embeddings using Dashscope
def generate_embeddings(texts):
    # Process in batches to avoid API limits
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = embedding_client.embeddings.create(
            model=model_name,
            input=batch,
            dimensions=embedding_dim
        )
        all_embeddings.extend([data.embedding for data in resp.data])
    return all_embeddings

# Generate embeddings for all chunks
print("Generating embeddings...")
embeddings = generate_embeddings([data["document"] for data in payload])
print(f"Generated {len(embeddings)} embeddings with {embedding_dim} dimensions")

# Check if collection already exists and handle dimension mismatch
try:
    # Try to get the collection info
    collection_info = client.get_collection(collection_name)
    print(f"Collection '{collection_name}' exists with {collection_info.config.params.vectors.size} dimensions")
    
    # Check if dimensions match
    if collection_info.config.params.vectors.size != embedding_dim:
        print(f"Dimension mismatch: expected {embedding_dim}, got {collection_info.config.params.vectors.size}")
        print("Deleting existing collection and recreating with correct dimensions...")
        client.delete_collection(collection_name=collection_name)
        
        # Recreate collection with correct dimensions
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim, distance=models.Distance.COSINE)
        )
    else:
        print(f"Collection '{collection_name}' already exists with correct dimensions. Skipping creation.")
        
except Exception as e:
    # Collection doesn't exist, so create it
    print(f"Collection '{collection_name}' does not exist. Creating it now.")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_dim, distance=models.Distance.COSINE)
    )

# Upload points with pre-computed embeddings
points = [
    models.PointStruct(
        id=ids[i],
        vector=embeddings[i],
        payload=payload[i]
    )
    for i in range(len(chunks))
]

client.upsert(
    collection_name=collection_name,
    points=points
)

# Perform search with embedding
query_text = "How to install this project?"
query_embedding = generate_embeddings([query_text])[0]

search_result = client.query_points(
    collection_name=collection_name,
    query=query_embedding
).points

print(f"Found {len(search_result)} results:")
for result in search_result[:3]:  # Show top 3 results
    print(f"ID: {result.id}, Score: {result.score}")
    print(f"Text: {result.payload.get('document', '')[:200]}...")
    print("---")