# Python Natural Language Processing Cookbook — Notebook Companion

Comprehensive, executable Jupyter notebooks for every chapter of **Python Natural Language Processing Cookbook, Second Edition** (Packt Publishing). Each notebook is self-contained, runs on Google Colab with a single click, and renders cleanly on GitHub.

> **Book repository:** [PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition](https://github.com/PacktPublishing/Python-Natural-LanguageProcessing-Cookbook-Second-Edition)

---

## What This Is

The original cookbook provides code spread across many standalone scripts and utility files. These notebooks **consolidate** each chapter into a single `.ipynb` with:

- All dependencies installed inline (no external utility notebooks required)
- Rich **Lead Data Scientist** explanations after every code output — citing exact numbers, deriving quantitative insights, and connecting results to production trade-offs
- LaTeX equations wherever the chapter introduces mathematical concepts
- Google Colab compatibility (GPU runtime, `pip install -q`, Colab Secrets for API keys)
- GitHub-clean rendering (no widget metadata, no ANSI escape codes, complete `language_info`)

---

## Chapters

| # | Chapter | Recipes | Key Topics |
|---|---------|---------|------------|
| 1 | Learning NLP Basics | — | Tokenization, stemming, lemmatization, POS tagging, stopwords |
| 2 | Playing with Grammar | — | Dependency parsing, noun chunks, subject/object extraction, pattern matching |
| 3 | Representing Text | — | Bag-of-words, TF-IDF, word embeddings, BERT/OpenAI embeddings, cosine similarity |
| 4 | [Classifying Texts](Chapter_04_Classifying_Texts.ipynb) | 6 | Rule-based, K-Means clustering, SVM + BERT, spaCy textcat, OpenAI zero-shot |
| 5 | [Information Extraction](Chapter_05_Information_Extraction.ipynb) | 6 | Regex, Levenshtein distance, TF-IDF keywords, spaCy NER, custom NER, BERT NER fine-tuning |
| 6 | [Topic Modeling](Chapter_06_Topic_Modeling.ipynb) | 5 | LDA (Gensim), SBERT community detection, K-Means + BERT, BERTopic, cross-lingual CTM |
| 7 | [Visualizing Text Data](Chapter_07_Visualizing_Text_Data.ipynb) | 7 | displaCy (dep parse, NER), POS bar charts, confusion matrices, word clouds, pyLDAvis, BERTopic viz |
| 8 | [Transformers and Their Applications](Chapter_08_Transformers.ipynb) | 6 | Datasets, tokenization, RoBERTa classification, zero-shot BART, GPT-2 generation, T5 translation |
| 9 | [Natural Language Understanding](Chapter_09_NLU.ipynb) | 8 | Extractive/abstractive QA, document-corpus QA, summarization (T5/BART/PEGASUS), entailment, LIME, Anchors |
| 10 | [Generative AI and LLMs](Chapter_10_GenAI_LLMs.ipynb) | 8 | Local LLMs (Mistral/Llama), instruction following, LangChain, RAG + FAISS, chatbot, code gen, text-to-SQL, ReAct agents |

Chapters 1–3 cover foundational NLP operations that are prerequisites for later chapters. Notebooks for Chapters 4–10 are complete and available above.

---

## Quick Start

### Option A — Google Colab (recommended)

1. Click any notebook link in the table above
2. In GitHub, click **Open in Colab** (or upload the `.ipynb` to [colab.research.google.com](https://colab.research.google.com))
3. Set the runtime to **T4 GPU** for Chapters 8–10 (`Runtime → Change runtime type → T4 GPU`)
4. Add your API keys in Colab Secrets (key icon in the left sidebar):
   - `HF_TOKEN` — Hugging Face access token
   - `OPENAI_API_KEY` — OpenAI API key
5. Run all cells (`Runtime → Run all`)

### Option B — Local Jupyter

```bash
pip install notebook
jupyter notebook Chapter_04_Classifying_Texts.ipynb
```

Most chapters work on CPU. Chapters 8–10 benefit significantly from a GPU (CUDA-compatible, 8+ GB VRAM).

---

## API Keys Required

| Key | Where to get it | Used in |
|-----|----------------|---------|
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Chapters 8, 9, 10 (gated models) |
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Chapters 4, 9, 10 (GPT-4o-mini) |

Keys are loaded via `google.colab.userdata.get()` — never hardcoded in notebook cells.

**Gated model access:** Chapters 9–10 use models that require approval on Hugging Face. Visit the model pages and request access before running:
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)

If access is not yet granted, the notebooks include notes on using ungated alternatives (e.g., `Qwen/Qwen2.5-7B-Instruct`).

---

## Notebook Design Principles

### Explanation Methodology — 6 Layers

Every code cell that produces output is followed by a markdown explanation applying up to six layers of analysis:

1. **State exact output values** — precise numbers, never approximated
2. **Interpret what the numbers mean** — why these results, not just what they are
3. **Include equations with intuition** — symbol-by-symbol explanation, not bare formulas
4. **Derive quantitative insights** — sparsity percentages, memory estimates, compression ratios
5. **Strategic/production insights** — what this means for a team, a product, a deployment
6. **Cross-chapter connections** — link to concepts from earlier or later chapters

### Technical Compatibility

Each notebook includes fixes for common Colab/GitHub issues:

- `HF_HUB_DISABLE_PROGRESS_BARS=1` to suppress download widgets
- `jupyter_client.session.utcnow` patch for the `datetime.utcnow()` deprecation warning
- No `metadata.widgets` (causes GitHub rendering failures)
- No ANSI escape codes in outputs
- Complete `language_info` and `kernelspec` metadata

---

## Technology Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| spaCy | 3.x | Tokenization, POS, NER, dependency parsing |
| NLTK | 3.x | Stopwords, tokenization, stemming |
| scikit-learn | 1.x | Classification (SVM, K-Means), evaluation metrics, TF-IDF |
| transformers | 4.x | BERT, RoBERTa, T5, GPT-2, Llama, Mistral |
| sentence-transformers | 2.x | Sentence embeddings (all-MiniLM-L6-v2, all-mpnet-base-v2) |
| LangChain | 0.3.x | Chains, RAG, agents, prompt templates |
| BERTopic | 0.x | Neural topic modeling |
| gensim | 4.x | LDA topic modeling |

### LLM Models Used

| Model | Parameters | Quantization | Recipes |
|-------|-----------|--------------|---------|
| Mistral-7B-v0.3 | 7.2B | 4-bit NF4 | 10.1 (local generation) |
| Llama-3.1-8B-Instruct | 8.0B | 4-bit NF4 | 10.2, 10.3 (instruction following) |
| GPT-4o-mini | — (API) | — | 10.4–10.8 (RAG, chatbot, code, SQL, agents) |
| BERT-large-uncased | 335M | FP32 | 9.1–9.4 (QA) |
| Flan-T5-base | 248M | FP32 | 9.5–9.6 (summarization, entailment) |

---

## Datasets

| Dataset | Source | Chapters |
|---------|--------|----------|
| BBC News (5 topics) | [SetFit/bbc-news](https://huggingface.co/datasets/SetFit/bbc-news) | 4, 6, 7 |
| Rotten Tomatoes | [rotten_tomatoes](https://huggingface.co/datasets/rotten_tomatoes) | 4, 8 |
| CoNLL-2003 (NER) | [conll2003](https://huggingface.co/datasets/conll2003) | 5 |
| 2024 Summer Olympics | Wikipedia (live scrape) | 10 |
| LLM Agents blog post | [lilianweng.github.io](https://lilianweng.github.io/posts/2023-06-23-agent/) | 10 |
| Seven Wonders corpus | Inline (7 documents) | 9 |
| Northwind-style DB | Inline SQLite (9 employees) | 10 |

---

## Notebook Statistics

| Chapter | Cells | Code | Markdown | Equations | Words |
|---------|-------|------|----------|-----------|-------|
| 4 — Classifying Texts | 78 | 38 | 40 | ~90 | ~7,500 |
| 5 — Information Extraction | 61 | 30 | 31 | ~65 | ~5,200 |
| 6 — Topic Modeling | 51 | 24 | 27 | ~50 | ~4,800 |
| 7 — Visualizing Text Data | 46 | 20 | 26 | ~30 | ~3,600 |
| 8 — Transformers | 42 | 19 | 23 | ~45 | ~3,900 |
| 9 — NLU | 56 | 24 | 32 | ~53 | ~3,100 |
| 10 — Generative AI & LLMs | 73 | 32 | 41 | ~76 | ~3,500 |
| **Total** | **407** | **187** | **220** | **~409** | **~31,600** |

---

## Known Issues and Workarounds

| Issue | Cause | Fix |
|-------|-------|-----|
| `GatedRepoError` for Llama/Mistral | Model access not yet approved | Request access on HF, or swap to `Qwen/Qwen2.5-7B-Instruct` |
| `torch_dtype is deprecated` | Newer transformers API change | Use `dtype` instead of `torch_dtype` (cosmetic warning, does not affect results) |
| `generation_config` deprecation | Conflicting generation params | Pass all params via `GenerationConfig` object (cosmetic warning) |
| PEGASUS gibberish output | Safetensors conversion crash | Set `HF_TOKEN`, use `torch.float16`, or try `google/pegasus-xsum` |
| BM25 retriever returning wrong docs | Lexical mismatch (synonym gap) | Use hybrid search (BM25 + dense retrieval) for production QA |

---

## Citation

If you use these notebooks in your teaching or research:

```
Zhenya Antić, "Python Natural Language Processing Cookbook, Second Edition"
Packt Publishing, 2024
```

---

## License

The notebook explanations and enhancements are provided for educational purposes. The original code recipes and datasets retain their respective licenses as specified by the book publisher and dataset authors. The BBC News dataset is used with permission from the original researchers (Greene & Cunningham, ICML 2006).
