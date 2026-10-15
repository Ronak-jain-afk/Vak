import collections
import hashlib
import re
from typing import Dict, Generator, Iterator, List, Optional, Set, Tuple
from datasketch import MinHash, MinHashLSH


def get_document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_minhash(text: str, num_perm: int = 128, shingle_size: int = 3) -> MinHash:
    # 3-word shingles capture phrasing and local syntax while detecting near-duplicate documents
    words = re.findall(r"\w+", text.lower())
    m = MinHash(num_perm=num_perm)
    if len(words) < shingle_size:
        m.update(" ".join(words).encode("utf-8"))
    else:
        shingles = [" ".join(words[i : i + shingle_size]).encode("utf-8") for i in range(len(words) - shingle_size + 1)]
        m.update_batch(shingles)
    return m


class Deduplicator:
    def __init__(self, threshold: float = 0.70, num_perm: int = 128, max_lsh_window: int = 50000):
        self.threshold = threshold
        self.num_perm = num_perm
        self.max_lsh_window = max_lsh_window
        self.seen_exact_hashes: Set[bytes] = set()
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.lsh_window = collections.deque()
        self.doc_count = 0
        self.exact_dupes = 0
        self.near_dupes = 0

    def is_duplicate(self, text: str, doc_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        self.doc_count += 1
        doc_key = doc_id if doc_id is not None else f"doc_{self.doc_count}"

        # 1. Exact match check (compact 64-bit digest to bound memory across 1.6M+ documents)
        exact_h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        if exact_h in self.seen_exact_hashes:
            self.exact_dupes += 1
            return True, "exact"
        self.seen_exact_hashes.add(exact_h)

        # 2. Near-duplicate check with bounded-memory MinHash LSH
        mh = compute_minhash(text, num_perm=self.num_perm)
        matches = self.lsh.query(mh)
        if matches:
            self.near_dupes += 1
            return True, f"near_match_with_{matches[0]}"

        # Evict oldest entry if window is filled to prevent memory exhaustion
        if self.max_lsh_window > 0 and len(self.lsh_window) >= self.max_lsh_window:
            oldest_key = self.lsh_window.popleft()
            self.lsh.remove(oldest_key)

        self.lsh.insert(doc_key, mh)
        if self.max_lsh_window > 0:
            self.lsh_window.append(doc_key)
        return False, None

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_checked": self.doc_count,
            "exact_duplicates": self.exact_dupes,
            "near_duplicates": self.near_dupes,
            "unique_documents": len(self.seen_exact_hashes) - self.near_dupes,
        }
