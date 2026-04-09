"""
Fuzzy Translation Memory Matcher
----------------------------------
Given a new source segment, finds the most similar segments in an existing
Translation Memory (TM) and suggests the corresponding translations.

How it works:
  1. Converts all TM source segments and the new query into TF-IDF vectors.
     TF-IDF (Term Frequency–Inverse Document Frequency) measures how important
     a word is within a document relative to the whole corpus.
  2. Computes cosine similarity between the query vector and every TM entry.
     Cosine similarity measures the angle between two vectors — closer to 1.0
     means more similar, 0.0 means completely different.
  3. Returns the top matches ranked by similarity score, just like a real CAT tool.

This is the same core logic used by professional tools like Trados and MemoQ —
this project makes the underlying NLP visible.

Author : Talha Umut Kulu
GitHub : https://github.com/tumut-k
"""

import math
import re
from collections import Counter


# ── TF-IDF implementation (no external libraries needed) ─────────────────────

def tokenize(text: str) -> list:
    """Lowercase and split into word tokens, removing punctuation."""
    return re.findall(r'\b[a-zA-ZğüşıöçĞÜŞİÖÇ]+\b', text.lower())


def compute_tf(tokens: list) -> dict:
    """Term Frequency: how often each word appears in this segment."""
    count = Counter(tokens)
    total = len(tokens)
    return {word: freq / total for word, freq in count.items()}


def compute_idf(corpus: list) -> dict:
    """
    Inverse Document Frequency: words that appear in every segment carry
    less discriminating power than words that appear in only a few.
    """
    num_docs = len(corpus)
    idf = {}
    all_words = set(word for doc in corpus for word in doc)
    for word in all_words:
        docs_containing = sum(1 for doc in corpus if word in doc)
        idf[word] = math.log(num_docs / (1 + docs_containing)) + 1
    return idf


def compute_tfidf_vector(tf: dict, idf: dict) -> dict:
    """Multiply TF and IDF to get the final weight for each word."""
    return {word: tf_val * idf.get(word, 1) for word, tf_val in tf.items()}


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """
    Cosine similarity between two TF-IDF vectors.
    Returns a float between 0.0 (no overlap) and 1.0 (identical).
    """
    # Dot product: sum of products of shared word weights
    shared_words = set(vec_a.keys()) & set(vec_b.keys())
    dot_product = sum(vec_a[w] * vec_b[w] for w in shared_words)

    # Magnitudes
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product / (mag_a * mag_b)


# ── Translation Memory class ──────────────────────────────────────────────────

class TranslationMemory:
    """
    A simple Translation Memory that stores (source, target) segment pairs
    and retrieves the closest matches for a new query segment.
    """

    def __init__(self):
        self.entries = []          # list of {"source": str, "target": str}
        self._vectors = []         # TF-IDF vectors for each source segment
        self._idf = {}             # IDF computed over the whole TM

    def add(self, source: str, target: str) -> None:
        """Add a (source, translation) pair to the TM."""
        self.entries.append({"source": source, "target": target})
        self._rebuild_index()

    def load_bulk(self, pairs: list) -> None:
        """Load a list of (source, target) tuples at once."""
        for source, target in pairs:
            self.entries.append({"source": source, "target": target})
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Recompute TF-IDF vectors whenever the TM changes."""
        tokenized = [tokenize(e["source"]) for e in self.entries]
        self._idf = compute_idf(tokenized)
        self._vectors = []
        for tokens in tokenized:
            tf = compute_tf(tokens)
            self._vectors.append(compute_tfidf_vector(tf, self._idf))

    def query(self, new_segment: str, top_n: int = 3, threshold: float = 0.0) -> list:
        """
        Find the most similar TM entries for a new source segment.

        Parameters
        ----------
        new_segment : the untranslated text to look up
        top_n       : how many matches to return
        threshold   : minimum similarity score to include (0.0 = all)

        Returns
        -------
        list of dicts: [{source, target, similarity, match_pct}]
        """
        tokens = tokenize(new_segment)
        tf = compute_tf(tokens)
        query_vec = compute_tfidf_vector(tf, self._idf)

        scored = []
        for i, entry in enumerate(self.entries):
            sim = cosine_similarity(query_vec, self._vectors[i])
            if sim >= threshold:
                scored.append({
                    "source":     entry["source"],
                    "target":     entry["target"],
                    "similarity": round(sim, 4),
                    "match_pct":  f"{round(sim * 100)}%",
                })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_n]


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_matches(query: str, matches: list) -> None:
    print("\n" + "═" * 62)
    print(f"  Query : {query}")
    print("═" * 62)
    if not matches:
        print("  No matches found above threshold.")
    for i, m in enumerate(matches, 1):
        bar_len = int(m["similarity"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"\n  Match #{i}  [{bar}] {m['match_pct']}")
        print(f"  Source : {m['source']}")
        print(f"  Target : {m['target']}")
    print()


# ── Sample Translation Memory (English → Turkish) ────────────────────────────
# Domain: medical interpreting — drawn from real terminology used in
# remote medical interpretation sessions.

SAMPLE_TM = [
    # Symptoms
    ("The patient reports severe chest pain.",
     "Hasta şiddetli göğüs ağrısı bildiriyor."),
    ("The patient has a high fever and chills.",
     "Hastanın yüksek ateşi ve titremesi var."),
    ("The patient is experiencing shortness of breath.",
     "Hasta nefes darlığı yaşıyor."),
    ("The patient reports nausea and vomiting.",
     "Hasta bulantı ve kusma şikayetinden bahsediyor."),
    ("There is swelling in the right ankle.",
     "Sağ ayak bilğinde şişlik var."),
    # Diagnoses
    ("The diagnosis is acute bronchitis.",
     "Tanı akut bronşit."),
    ("The test result came back negative.",
     "Test sonucu negatif çıktı."),
    ("Blood pressure is within normal range.",
     "Kan basıncı normal sınırlar içinde."),
    # Instructions
    ("Please take this medication twice a day with food.",
     "Lütfen bu ilacı günde iki kez yemekle birlikte alın."),
    ("You need to rest and drink plenty of fluids.",
     "Dinlenmeniz ve bol sıvı içmeniz gerekiyor."),
    ("Your follow-up appointment is in two weeks.",
     "Kontrol randevunuz iki hafta sonra."),
    # Consent / legal
    ("Do you consent to this procedure?",
     "Bu işleme onay veriyor musunuz?"),
    ("Please sign the informed consent form.",
     "Lütfen bilgilendirilmiş onam formunu imzalayın."),
]


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_demo():
    print("\nFuzzy Translation Memory Matcher — Demo")
    print("Loading translation memory...")

    tm = TranslationMemory()
    tm.load_bulk(SAMPLE_TM)
    print(f"TM loaded: {len(tm.entries)} segments\n")

    test_queries = [
        "The patient is suffering from chest pain.",      # close to entry 1
        "She has fever.",                                 # partial match entry 2
        "The test result was negative.",                  # near match entry 7
        "Please sign the consent form.",                  # near match entry 13
        "The weather is nice today.",                     # low similarity — no match
    ]

    for query in test_queries:
        matches = tm.query(query, top_n=2, threshold=0.1)
        print_matches(query, matches)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Query mode: python tm_matcher.py "your segment here"
        tm = TranslationMemory()
        tm.load_bulk(SAMPLE_TM)
        query = " ".join(sys.argv[1:])
        matches = tm.query(query, top_n=3, threshold=0.05)
        print_matches(query, matches)
    else:
        run_demo()
