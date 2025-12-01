import requests
import asyncio
import aiohttp
import logging
from typing import List, Dict


class RetrievalClient:
    """Simple client for the retrieval service"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def retrieve(self, query: str, top_k: int = None) -> Dict:
        """Synchronous retrieval"""
        url = f"{self.base_url}/retrieve"
        payload = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def batch_retrieve(self, queries: List[str], top_k: int = None) -> Dict:
        """Synchronous batch retrieval"""
        url = f"{self.base_url}/batch_retrieve"
        payload = {"queries": queries}
        if top_k is not None:
            payload["top_k"] = top_k

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def retrieve_async(self, query: str, top_k: int = None) -> Dict:
        """Async retrieval for high concurrency"""
        url = f"{self.base_url}/retrieve"
        payload = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    async def batch_retrieve_async(self,
                                   queries: List[str],
                                   top_k: int = None) -> Dict:
        """Async batch retrieval"""
        url = f"{self.base_url}/batch_retrieve"
        payload = {"queries": queries}
        if top_k is not None:
            payload["top_k"] = top_k

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    async def concurrent_retrieve(self,
                                  queries: List[str],
                                  top_k: int = None) -> List[Dict]:
        """Retrieve multiple queries concurrently"""
        tasks = [self.retrieve_async(q, top_k) for q in queries]
        return await asyncio.gather(*tasks)

    def health_check(self) -> Dict:
        """Check service health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


# Initialize client once (reuse across requests)
client = RetrievalClient("http://localhost:8008")


def retrieve_chuncks(query: str, top_k: int = 3) -> list[str]:
    """
    Synchronous RPC function - thread-safe
    Can be called concurrently from multiple threads
    """
    print(query)
    context = []
    try:
        results = client.retrieve(query, top_k=top_k)
        for i, r in enumerate(results['results'], 1):
            context.append(r['text'])
    except Exception as e:
        logging.error(
            f"Retrieval Failed. Search Query is: {query}. Error: {e}")

    return context


if __name__ == "__main__":
    t = retrieve_chuncks("Hello World", 6)
    print(t)
    print(len(t))
