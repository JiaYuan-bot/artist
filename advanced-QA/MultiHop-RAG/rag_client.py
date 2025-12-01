import requests
import asyncio
import aiohttp
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


# Example usage
if __name__ == "__main__":
    client = RetrievalClient("http://localhost:8008")

    # 1. Health check
    print("Health Check:")
    print(client.health_check())
    print()

    # 2. Single query
    print("Single Query:")
    result = client.retrieve("What are the best deals on Amazon?", top_k=3)
    print(f"Query: {result['query']}")
    print(f"Found {result['count']} results:")
    for i, r in enumerate(result['results'], 1):
        print(f"  [{i}] Score: {r['score']:.4f}")
        print(f"      Title: {r['metadata'].get('title', 'N/A')[:60]}")
        print()

    # 4. High concurrency example
    print("Concurrent Queries:")

    async def test_concurrent():
        queries = [f"Query {i}" for i in range(10)]
        results = await client.concurrent_retrieve(queries, top_k=2)
        print(f"Completed {len(results)} concurrent requests")
        return results

    asyncio.run(test_concurrent())
