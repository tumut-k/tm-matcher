# Fuzzy Translation Memory Matcher

A from-scratch implementation of the core algorithm behind professional CAT (Computer-Assisted Translation) tools like Trados and MemoQ — **fuzzy matching against a Translation Memory**.

No external NLP libraries required. Built entirely with Python's standard library.

---

## What is a Translation Memory?

A Translation Memory (TM) is a database that stores previously translated segment pairs:

```
Source (EN): "The patient has a high fever."
Target (TR): "Hastanın yüksek ateşi var."
```

When a new segment arrives that is *similar but not identical* to a stored one, the tool suggests the closest match — saving translators from re-translating near-duplicate content. This is called a **fuzzy match**.

---

## How this works — the NLP pipeline

```
New segment → Tokenize → TF-IDF vector → Cosine similarity → Ranked matches
```

**Step 1 — Tokenization**
The input text is split into word tokens (punctuation removed, lowercased).

**Step 2 — TF-IDF vectorization**
Each segment is converted into a numerical vector where each dimension represents a word. The weight of each word is its TF-IDF score:
- **TF (Term Frequency):** how often the word appears in this segment
- **IDF (Inverse Document Frequency):** words that appear in every segment (like "the", "a") get down-weighted; rare domain-specific words (like "bronchitis") get up-weighted

**Step 3 — Cosine similarity**
Measures the angle between two TF-IDF vectors. A score of **1.0** means identical content; **0.0** means no shared vocabulary.

```
similarity("The patient has chest pain", "The patient reports chest pain")
→ 0.87  (87% match)
```

**Step 4 — Ranking**
All TM entries are scored against the query and returned sorted by similarity, just like a real CAT tool's match report.

---

## Example output

```
══════════════════════════════════════════════════════════════
  Query : The patient is suffering from chest pain.
══════════════════════════════════════════════════════════════

  Match #1  [████████████████░░░░] 82%
  Source : The patient reports severe chest pain.
  Target : Hasta şiddetli göğüs ağrısı bildiriyor.

  Match #2  [████████░░░░░░░░░░░░] 41%
  Source : The patient has a high fever and chills.
  Target : Hastanın yüksek ateşi ve titremesi var.
```

---

## How to run

**Option 1 — Google Colab**
1. Copy the full contents of `tm_matcher.py` into a Colab cell
2. Run the cell — no installation needed
3. Try your own queries: `tm.query("your sentence here", top_n=3)`

**Option 2 — Local**
```bash
git clone https://github.com/tumut-k/tm-matcher
cd tm-matcher

# Run the built-in demo
python tm_matcher.py

# Query with your own text
python tm_matcher.py "The patient needs to sign the consent form."
```

---

## Sample Translation Memory

The built-in TM contains 13 English→Turkish segment pairs from the medical interpreting domain — drawn from real terminology I use in remote medical interpretation sessions (ICE certification, HIPAA-compliant).

| Domain | EN example | TR example |
|--------|-----------|-----------|
| Symptoms | The patient reports nausea and vomiting. | Hasta bulantı ve kusma şikayetinden bahsediyor. |
| Diagnosis | The test result came back negative. | Test sonucu negatif çıktı. |
| Instructions | Please take this medication twice a day. | Lütfen bu ilacı günde iki kez alın. |
| Consent | Do you consent to this procedure? | Bu işleme onay veriyor musunuz? |

---

## Limitations & next steps

- Current implementation uses **bag-of-words** TF-IDF — word order is not preserved
- A production version would use **sentence embeddings** (e.g., `sentence-transformers`) for semantic similarity that handles paraphrases better
- Planned: bilingual TF-IDF so the matcher can query by *target language* segment too
- Planned: load TM from standard `.tmx` files (the industry format used by Trados, MemoQ, etc.)

---

## Why I built this

I have worked as a professional translator and medical interpreter for several years. CAT tools are central to that workflow, but their matching algorithms are a black box to most linguists. Building this from scratch — implementing TF-IDF and cosine similarity by hand — made the underlying NLP concrete and visible.

This project sits at the intersection of my translation background and my current NLP work at [LILT](https://lilt.com), where I evaluate neural machine translation outputs and work with large-scale localization pipelines.

---

## About

**Talha Umut Kulu** — professional translator (500,000+ words, medical/legal/technical) and Turkish AI Content Expert at LILT, currently transitioning into NLP research.

- GitHub: [github.com/tumut-k](https://github.com/tumut-k)
- LinkedIn: [linkedin.com/in/tumutk](https://linkedin.com/in/tumutk)
