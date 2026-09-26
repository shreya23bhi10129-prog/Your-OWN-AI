"""
VectorDB Engine 

Requirements:
    pip install flask requests

Run:
    python vectordb_server.py
    # then open http://localhost:8080  (serves ./index.html if present)

Ollama (optional, needed for the /doc/* RAG endpoints):
    https://ollama.com
    ollama pull nomic-embed-text
    ollama pull llama3.2
"""

import heapq
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, request, send_file

DIMS = 16  # demo vector dimensionality


# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]


DistFn = Callable[[List[float], List[float]], float]


# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)


def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(name: str) -> DistFn:
    if name == "cosine":
        return cosine
    if name == "manhattan":
        return manhattan
    return euclidean


# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        r = [(dist(q, v.emb), v.id) for v in self.items]
        r.sort(key=lambda x: x[0])
        return r[:k]

    def remove(self, item_id: int) -> None:
        self.items = [v for v in self.items if v.id != item_id]


# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None


class KDTree:
    def __init__(self, dims: int):
        self.root: Optional[KDNode] = None
        self.dims = dims

    def _ins(self, node: Optional[KDNode], v: VectorItem, d: int) -> KDNode:
        if node is None:
            return KDNode(v)
        ax = d % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left = self._ins(node.left, v, d + 1)
        else:
            node.right = self._ins(node.right, v, d + 1)
        return node

    def insert(self, v: VectorItem) -> None:
        self.root = self._ins(self.root, v, 0)

    def _knn(self, node: Optional[KDNode], q: List[float], k: int, d: int,
             dist: DistFn, heap: List[Tuple[float, int, int]], counter: List[int]) -> None:
        # heap is a max-heap of (-dist, tiebreaker, id) implemented via negated distances
        if node is None:
            return
        dn = dist(q, node.item.emb)
        if len(heap) < k:
            counter[0] += 1
            heapq.heappush(heap, (-dn, counter[0], node.item.id))
        elif dn < -heap[0][0]:
            counter[0] += 1
            heapq.heapreplace(heap, (-dn, counter[0], node.item.id))

        ax = d % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer, farther = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._knn(closer, q, k, d + 1, dist, heap, counter)
        worst = -heap[0][0] if len(heap) >= k else float("inf")
        if len(heap) < k or abs(diff) < worst:
            self._knn(farther, q, k, d + 1, dist, heap, counter)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        heap: List[Tuple[float, int, int]] = []
        counter = [0]
        self._knn(self.root, q, k, 0, dist, heap, counter)
        r = [(-nd, i) for (nd, _, i) in heap]
        r.sort(key=lambda x: x[0])
        return r

    def rebuild(self, items: List[VectorItem]) -> None:
        self.root = None
        for v in items:
            self.insert(v)


# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSWNode:
    __slots__ = ("item", "max_lyr", "nbrs")

    def __init__(self, item: VectorItem, max_lyr: int):
        self.item = item
        self.max_lyr = max_lyr
        self.nbrs: List[List[int]] = [[] for _ in range(max_lyr + 1)]


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200):
        self.G: Dict[int, HNSWNode] = {}
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(m)
        self.top_layer = -1
        self.entry_pt = -1
        self._rng = random.Random(42)

    def _rand_level(self) -> int:
        u = self._rng.random()
        while u <= 0.0:
            u = self._rng.random()
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(self, q: List[float], ep: int, ef: int, lyr: int,
                       dist: DistFn) -> List[Tuple[float, int]]:
        visited = {ep}
        d0 = dist(q, self.G[ep].item.emb)
        # cands: min-heap of (dist, id)
        cands: List[Tuple[float, int]] = [(d0, ep)]
        heapq.heapify(cands)
        # found: max-heap of (dist, id) via negation
        found: List[Tuple[float, int]] = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = self.G.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in visited or nid not in self.G:
                    continue
                visited.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        res = [(-fd, fid) for fd, fid in found]
        res.sort(key=lambda x: x[0])
        return res

    @staticmethod
    def _select_nbrs(cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [cid for _, cid in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        item_id = item.id
        lvl = self._rand_level()
        self.G[item_id] = HNSWNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = item_id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                w = self._search_layer(item.emb, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            w = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(w, max_m)
            self.G[item_id].nbrs[lc] = sel

            for nid in sel:
                nnode = self.G.get(nid)
                if nnode is None:
                    continue
                if len(nnode.nbrs) <= lc:
                    nnode.nbrs.extend([[] for _ in range(lc + 1 - len(nnode.nbrs))])
                conn = nnode.nbrs[lc]
                conn.append(item_id)
                if len(conn) > max_m:
                    ds = [(dist(nnode.item.emb, self.G[c].item.emb), c)
                          for c in conn if c in self.G]
                    ds.sort(key=lambda x: x[0])
                    nnode.nbrs[lc] = [c for _, c in ds[:max_m]]

            if w:
                ep = w[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = item_id

    def knn(self, q: List[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                w = self._search_layer(q, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]
        w = self._search_layer(q, ep, max(ef, k), 0, dist)
        return w[:k]

    def remove(self, item_id: int) -> None:
        if item_id not in self.G:
            return
        for node in self.G.values():
            for layer in node.nbrs:
                if item_id in layer:
                    layer.remove(item_id)
        if self.entry_pt == item_id:
            self.entry_pt = -1
            for other_id in self.G:
                if other_id != item_id:
                    self.entry_pt = other_id
                    break
        del self.G[item_id]

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []
        for node_id, node in self.G.items():
            nodes.append({
                "id": node_id,
                "metadata": node.item.metadata,
                "category": node.item.category,
                "maxLyr": node.max_lyr,
            })
            for lc in range(min(node.max_lyr, max_l - 1) + 1):
                nodes_per_layer[lc] += 1
                if lc < len(node.nbrs):
                    for nid in node.nbrs[lc]:
                        if node_id < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": node_id, "dst": nid, "lyr": lc})
        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def size(self) -> int:
        return len(self.G)


# =====================================================================
#  VECTOR DATABASE (demo fixed-dim index)
# =====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.mu = threading.Lock()
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: List[float], dist: DistFn) -> int:
        with self.mu:
            v = VectorItem(self.next_id, meta, cat, emb)
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
                algo = "hnsw"

            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, item_id in raw:
                v = self.store.get(item_id)
                if v:
                    hits.append({
                        "id": v.id,
                        "metadata": v.metadata,
                        "category": v.category,
                        "distance": d,
                        "embedding": v.emb,
                    })
            return {"hits": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, q: List[float], k: int, metric: str) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)

            def timed(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            bf_us = timed(lambda: self.bf.knn(q, k, dfn))
            kd_us = timed(lambda: self.kdt.knn(q, k, dfn))
            hnsw_us = timed(lambda: self.hnsw.knn(q, k, 50, dfn))
            return {
                "bruteforceUs": bf_us,
                "kdtreeUs": kd_us,
                "hnswUs": hnsw_us,
                "itemCount": len(self.store),
            }

    def all(self) -> List[VectorItem]:
        with self.mu:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self.mu:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.mu:
            return len(self.store)


# =====================================================================
#  JSON / TEXT HELPERS
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks


# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text: str) -> List[float]:
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except requests.RequestException:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"


# =====================================================================
#  DOCUMENT DATABASE — HNSW over real Ollama embeddings
# =====================================================================

class DocumentDB:
    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.mu:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self.mu:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)
            out = []
            for d, item_id in raw:
                if item_id in self.store and d <= max_dist:
                    out.append((d, self.store[item_id]))
            return out

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.hnsw.remove(item_id)
            self.bf.remove(item_id)
            return True

    def all(self) -> List[DocItem]:
        with self.mu:
            return list(self.store.values())

    def size(self) -> int:
        with self.mu:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims


# =====================================================================
#  DEMO DATA (16D categorical vectors)
# =====================================================================

def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")
    # Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    demo = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]),
    ]
    for meta, cat, emb in demo:
        db.insert(meta, cat, emb, dist)


# =====================================================================
#  HTTP SERVER (Flask)
# =====================================================================

app = Flask(__name__)
db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/<path:_any>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def options_handler(_any=None):
    return ("", 204)


def parse_vec(s: str) -> List[float]:
    if not s:
        return []
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


# ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

@app.route("/search", methods=["GET"])
def search():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})
    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"
    out = db.search(q, k, metric, algo)
    return jsonify({
        "results": out["hits"],
        "latencyUs": out["latencyUs"],
        "algo": out["algo"],
        "metric": out["metric"],
    })


@app.route("/insert", methods=["POST"])
def insert():
    body = request.get_json(silent=True) or {}
    meta = body.get("metadata", "")
    cat = body.get("category", "")
    emb = body.get("embedding", [])
    if not meta or not emb or len(emb) != DIMS:
        return jsonify({"error": "invalid body"})
    item_id = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": item_id})


@app.route("/delete/<int:item_id>", methods=["DELETE"])
def delete(item_id):
    ok = db.remove(item_id)
    return jsonify({"ok": ok})


@app.route("/items", methods=["GET"])
def items():
    out = [{"id": v.id, "metadata": v.metadata, "category": v.category, "embedding": v.emb}
           for v in db.all()]
    return jsonify(out)


@app.route("/benchmark", methods=["GET"])
def benchmark():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})
    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5
    metric = request.args.get("metric") or "cosine"
    b = db.benchmark(q, k, metric)
    return jsonify(b)


@app.route("/hnsw-info", methods=["GET"])
def hnsw_info():
    return jsonify(db.hnsw_info())


# ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    body = request.get_json(silent=True) or {}
    title = body.get("title", "")
    text = body.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"})

    chunks = chunk_text(text, 250, 30)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({
                "error": "Ollama unavailable. Install from https://ollama.com then run: "
                         "ollama pull nomic-embed-text && ollama pull llama3.2"
            })
        chunk_title = f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})


@app.route("/doc/delete/<int:item_id>", methods=["DELETE"])
def doc_delete(item_id):
    ok = doc_db.remove(item_id)
    return jsonify({"ok": ok})


@app.route("/doc/list", methods=["GET"])
def doc_list():
    out = []
    for d in doc_db.all():
        preview = d.text[:120]
        if len(d.text) > 120:
            preview += "…"
        out.append({
            "id": d.id,
            "title": d.title,
            "preview": preview,
            "words": len(d.text.split()),
        })
    return jsonify(out)


@app.route("/doc/search", methods=["POST"])
def doc_search():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"})

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    hits = doc_db.search(q_emb, k)
    out = [{"id": d.id, "title": d.title, "distance": dist} for dist, d in hits]
    return jsonify({"contexts": out})


@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"})

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    hits = doc_db.search(q_emb, k)

    ctx = ""
    for i, (dist, d) in enumerate(hits):
        ctx += f"[{i + 1}] {d.title}:\n{d.text}\n\n"

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}Question: {question}\n\nAnswer:"
    )

    answer = ollama.generate(prompt)

    contexts = [{"id": d.id, "title": d.title, "text": d.text, "distance": dist}
                for dist, d in hits]

    return jsonify({
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": contexts,
        "docCount": doc_db.size(),
    })


# ── STATUS / STATS ────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    up = ollama.is_available()
    return jsonify({
        "ollamaAvailable": up,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size(),
    })


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    })


@app.route("/", methods=["GET"])
def index():
    try:
        return send_file("index.html")
    except FileNotFoundError:
        return ("", 404)


if __name__ == "__main__":
    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    app.run(host="0.0.0.0", port=8080, threaded=True)
