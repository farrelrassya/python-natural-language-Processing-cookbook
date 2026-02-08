# Learning NLP Basics

Hands-on notebooks covering fundamental Natural Language Processing (NLP) preprocessing techniques using **NLTK**, **spaCy**, and the **OpenAI API**.

## Notebooks

| # | Notebook | Topics |
|---|---|---|
| 1 | `Learning_NLP_01.ipynb` | Sentence splitting, Word tokenization |
| 2 | `Learning_NLP_02.ipynb` | POS tagging, Lemmatization, Stopword removal |

## What You'll Learn

**Part 1** — Text segmentation fundamentals:
- Divide text into sentences using NLTK Punkt tokenizer and spaCy
- Tokenize sentences into words (including multi-word expressions)
- Speed comparison between NLTK and spaCy approaches

**Part 2** — Deeper text analysis:
- Part-of-Speech tagging with spaCy (Universal Dependencies), NLTK (Penn Treebank), and OpenAI GPT
- Lemmatization to reduce words to their dictionary form
- Stopword removal using predefined lists and frequency-based methods
- Complete preprocessing pipeline combining all steps

## Requirements

```
nltk
spacy
openai     # optional, for GPT POS tagging section
matplotlib # for visualization
```

The notebooks auto-download required NLTK data and spaCy models (`en_core_web_sm`) on first run.

## Quick Start

These notebooks are designed to run on **Google Colab** — no local setup needed:

1. Upload the notebook to [Google Colab](https://colab.research.google.com/)
2. Run all cells from top to bottom
3. For the OpenAI section (Part 2), add your API key via Colab Secrets (`OPENAI_API_KEY`)

## Sample Text

Both notebooks use an excerpt from *The Adventures of Sherlock Holmes* by Arthur Conan Doyle as the running example. The text is embedded directly in the notebooks.

## References

- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/)
- [Universal Dependencies POS Tags](https://universaldependencies.org/u/pos/)
- [Punkt Tokenizer Paper](https://aclanthology.org/J06-4003.pdf)
- [spaCy Processing Pipelines](https://spacy.io/usage/processing-pipelines)

Based on: *Python Natural Language Processing Cookbook, Second Edition* (Packt Publishing)
