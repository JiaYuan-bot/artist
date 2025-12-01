# simple_retrieval_faiss_no_autorebuild.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Optional

import google.auth
from google.auth.transport.requests import Request

from llama_index.core import (VectorStoreIndex, Document, Settings,
                              StorageContext, load_index_from_storage)
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

import faiss


class SimpleRetrievalServiceFaiss:
    """Vertex AI embeddings + FAISS with persistence (no auto-rebuild)."""

    def __init__(
        self,
        corpus_path: str,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "text-embedding-004",
        chunk_size: int = 512,
        top_k: int = 5,
        persist_dir: str = "./li_store",
        faiss_index_path: str = "./faiss.index",
        distance: str = "ip",
    ):
        self.corpus_path = corpus_path
        self.top_k = top_k
        self.persist_dir = persist_dir
        self.faiss_index_path = faiss_index_path
        self.distance = distance.lower()

        # ---- Vertex AI embedding with ADC ----
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"])
        try:
            creds.refresh(Request())
        except Exception:
            pass

        Settings.embed_model = VertexTextEmbedding(model_name=model_name,
                                                   project=project_id,
                                                   location=location,
                                                   credentials=creds,
                                                   embed_batch_size=100)
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = 50

        # ---- Load if present, else build once ----
        self.index = self._load_or_build_index()
        print("Index ready.")

    # ---------- public ----------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        k = top_k if top_k is not None else self.top_k
        retriever = self.index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        return [{
            "text": n.get_content(),
            "score": n.get_score(),
            "metadata": n.metadata
        } for n in nodes]

    def batch_retrieve(self,
                       queries: List[str],
                       top_k: Optional[int] = None) -> List[List[Dict]]:
        return [self.retrieve(q, top_k) for q in queries]

    def rebuild(self) -> None:
        """Manual rebuild from corpus; overwrites persisted stores."""
        import shutil
        # 删除旧文件
        if Path(self.persist_dir).exists():
            shutil.rmtree(self.persist_dir)
        if Path(self.faiss_index_path).exists():
            Path(self.faiss_index_path).unlink()

        self.index = self._build_and_persist()
        print("Rebuilt FAISS + docstore from corpus.")

    # ---------- internals ----------
    def _load_or_build_index(self) -> VectorStoreIndex:
        pd = Path(self.persist_dir)
        fi = Path(self.faiss_index_path)

        if pd.exists() and fi.exists():
            print(f"Loading existing index from {self.persist_dir}...")
            try:
                # 从文件加载 FAISS 索引
                faiss_index = faiss.read_index(str(fi))
                vector_store = FaissVectorStore(faiss_index=faiss_index)

                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store, persist_dir=str(pd))
                return load_index_from_storage(storage_context)
            except Exception as e:
                print(f"Failed to load index: {e}")
                print("Building new index...")

        return self._build_and_persist()

    def _build_and_persist(self) -> VectorStoreIndex:
        print("Building new FAISS index...")
        docs = self._load_corpus()
        dim = self._probe_dim()
        print(f"Embedding dimension: {dim}")

        # 创建 FAISS 索引
        if self.distance == "ip":
            faiss_index = faiss.IndexFlatIP(dim)
        elif self.distance == "l2":
            faiss_index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError("distance must be 'ip' or 'l2'")

        # 创建 vector store
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        # 创建 storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)

        # 构建索引
        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, show_progress=True)

        # 持久化 LlamaIndex 元数据
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        storage_context.persist(persist_dir=self.persist_dir)

        # 持久化 FAISS 索引
        faiss.write_index(vector_store.client, self.faiss_index_path)
        print(f"Saved FAISS index to {self.faiss_index_path}")
        print(f"Saved docstore to {self.persist_dir}")

        return index

    def _load_corpus(self) -> List[Document]:
        print(f"📚 Loading corpus from {self.corpus_path}...")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs: List[Document] = []
        for item in data:
            docs.append(
                Document(
                    text=item.get("body", ""),
                    metadata={
                        "title": item.get("title", ""),
                        "source": item.get("source", ""),
                        "url": item.get("url", ""),
                        "published_at": item.get("published_at", ""),
                        "author": item.get("author", ""),
                        "category": item.get("category", ""),
                    },
                ))
        print(f"Loaded {len(docs)} documents")
        return docs

    def _probe_dim(self) -> int:
        """探测 embedding 维度"""
        try:
            vec = Settings.embed_model.get_text_embedding("dim probe")
        except Exception:
            vec = Settings.embed_model.get_query_embedding("dim probe")

        dim = len(vec)
        if dim <= 0:
            raise RuntimeError("Failed to probe embedding dimension.")
        return dim


# ---------- example ----------
if __name__ == "__main__":
    PROJECT_ID = "dbgroup"
    CORPUS_PATH = "/home/jiayuan/nl2sql/MultiHop-RAG/dataset/corpus.json"

    # 初始化服务
    svc = SimpleRetrievalServiceFaiss(
        corpus_path=CORPUS_PATH,
        project_id=PROJECT_ID,
        location="us-central1",
        model_name="gemini-embedding-001",
        chunk_size=256,
        top_k=5,
        persist_dir="./li_store",
        faiss_index_path="./faiss.index",
        distance="ip",
    )

    # 测试检索
    q = "What are the best deals on Amazon?"
    print(f"\n Query: {q}\n")

    results = svc.retrieve(q, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']:.4f}")
        print(f"    Title: {r['metadata'].get('title', 'N/A')[:60]}")
        print(f"    Source: {r['metadata'].get('source', 'N/A')}")
        print(f"    Text: {r['text'][:100]}...")
        print()
