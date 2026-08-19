# Corpus

Put public-source research documents here as Markdown files. Each file should
start with a YAML-style front-matter block:

```
---
title: ...
source_url: https://...
source_type: job_page | company | news | paper
collected_at: 2026-08-19
tags: [shopee, ai, internship]
---
```

The seed corpus is created by `python scripts/seed_corpus.py`.
