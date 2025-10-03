# Qdrant Setup Guide

This guide will help you set up Qdrant for use with the crawl4AI-agent-v2 project.

## Option 1: Local Installation with Docker (Recommended for Development)

1. Install Docker Desktop if you haven't already

2. Run Qdrant in a Docker container:
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

3. For Windows (Command Prompt):
```cmd
docker run -d -p 6333:6333 -p 6334:6334 ^
    -v %cd%/qdrant_storage:/qdrant/storage:z ^
    qdrant/qdrant
```

4. For Windows (PowerShell):
```powershell
docker run -d -p 6333:6333 -p 6334:6334 `
    -v ${PWD}/qdrant_storage:/qdrant/storage:z `
    qdrant/qdrant
```

5. Verify Qdrant is running:
- REST API: http://localhost:6333
- Web UI: http://localhost:6333/dashboard

## Option 2: Qdrant Cloud (Recommended for Production)

1. Sign up at https://cloud.qdrant.io/
2. Create a new cluster
3. Get your cluster URL and API key from the dashboard
4. Update your `.env` file with:
```env
QDRANT_URL=https://YOUR-CLUSTER-URL.qdrant.tech:6333
QDRANT_API_KEY=YOUR-API-KEY
```

## Configuration for crawl4AI-agent-v2

After setting up Qdrant, update your `.env` file:

For local Qdrant:
```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Leave empty for local instances
```

For Qdrant Cloud:
```env
QDRANT_URL=https://YOUR-CLUSTER-URL.qdrant.tech:6333
QDRANT_API_KEY=YOUR-API-KEY
```

## Testing Your Setup

You can test your Qdrant setup with a simple Python script:

```python
from qdrant_client import QdrantClient

# Initialize client
client = QdrantClient(url="http://localhost:6333")  # or your cloud URL

# List collections
collections = client.get_collections()
print("Collections:", collections)
```

## Troubleshooting

1. **Port already in use**: Make sure no other service is using ports 6333 or 6334
2. **Permission issues**: Ensure Docker has permission to access your file system
3. **Connection refused**: Check that the Docker container is running with `docker ps`
4. **Memory issues**: Qdrant may require more memory for large datasets

For more information, visit the [Qdrant documentation](https://qdrant.tech/documentation/).