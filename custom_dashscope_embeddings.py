#!/usr/bin/env python3
"""Custom DashScope embeddings that work with the compatible mode endpoint"""

import requests
from typing import List, Optional
from langchain_core.embeddings import Embeddings

class CustomDashScopeEmbeddings(Embeddings):
    """Custom DashScope embeddings that use the compatible mode endpoint"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-v4"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.model,
            "input": text
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code != 200:
            raise ValueError(f"API request failed: {response.status_code} - {response.text}")
        
        result = response.json()
        embedding = result.get('data', [{}])[0].get('embedding', [])
        
        if not embedding:
            raise ValueError("No embedding found in response")
            
        return embedding

# Test the custom embeddings
if __name__ == "__main__":
    api_key = "sk-4bec0483d7904c71b75a149f99471bd2"
    
    print("Testing custom DashScope embeddings...")
    embeddings = CustomDashScopeEmbeddings(api_key=api_key)
    
    # Test single embedding
    result = embeddings.embed_query("test")
    print(f"✓ Single embedding dimension: {len(result)}")
    
    # Test multiple embeddings
    results = embeddings.embed_documents(["test1", "test2"])
    print(f"✓ Multiple embeddings: {len(results)} documents, dimension: {len(results[0])}")
    
    print("✓ Custom embeddings working correctly!")