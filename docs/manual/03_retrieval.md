# 第三章 混合检索层

## 3.1 检索层组成

检索层位于 core/retrieval/，包含五个模块：

- chunker.py：读取 Markdown，解析 front matter，按标题和段落切块。
- bm25.py：无第三方依赖的 BM25 实现。
- embeddings.py：Hash、SentenceTransformer、FlagEmbedding 三种向量实现。
- rerank.py：Identity、CrossEncoder、FlagReranker 三种重排实现。
- index.py：组合 BM25、向量和重排，对外暴露 RetrievalIndex.search。

检索层的输入是 data/corpus 下的 Markdown 文件，输出是 RankedChunk 列表。AgentBrain 再把它转换成 Source。下图展示完整路径：

```text
data/corpus/**/*.md
  |
  v
load_markdown_corpus
  |
  v
chunk_markdown_document
  |
  v
Chunk(tokens, embedding)
  |
  +--> BM25Okapi
  +--> dense cosine
  |
  v
RetrievalIndex.search
  |
  v
RankedChunk
  |
  v
Source
```

## 3.2 分词器 chunker.tokenize

```python
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]
```

分词器同时支持英文、数字、下划线和中文。中文按连续汉字串作为一个 token。例如 "新加坡 AI" 会得到 ["新加坡", "ai"]，"中文混合abc" 会得到 ["中文混合", "abc"]。这个设计解决了一个真实问题：导入的岗位文档包含大量中文，如果只使用 ASCII 正则，中文内容会变成空 token，BM25 无法命中。

## 3.3 Front Matter 解析

parse_front_matter 解析 Markdown 文件开头的 YAML 风格元数据。

```python
---
title: Shopee AI Internship Opportunities in Singapore
source_url: https://careers.shopee.sg/jobs
source_type: job_page
collected_at: 2026-08-19
tags: [shopee, ai, internship]
---
```

函数返回 (metadata, body)。metadata 中的 title、source_url、tags 会进入每个 Chunk 的 metadata，最终用于报告引用。没有 front matter 时返回空 dict 和原文。

当前实现是简化解析器，不是完整 YAML。它支持 key: value 和 [a, b, c] 两种形式。复杂嵌套 YAML 不会被解析。

## 3.4 Chunk 生成算法

### 3.4.1 Chunk ID

```python
def _chunk_id(doc_id: str, text: str) -> str:
    digest = hashlib.sha1(f"{doc_id}:{text}".encode()).hexdigest()[:12]
    return f"{doc_id}-{digest}"
```

Chunk ID 由 doc_id 与正文的 SHA-1 前 12 位组成，保证同一文档同一内容的 chunk 稳定，不同内容不会碰撞到常见程度。

### 3.4.2 _make_chunk

```python
def _make_chunk(doc_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    clean = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    return Chunk(
        id=_chunk_id(doc_id, clean),
        doc_id=doc_id,
        text=clean,
        metadata=dict(metadata),
        tokens=tokenize(clean),
    )
```

_make_chunk 会去掉空行、复制 metadata、生成 tokens。注意 embedding 此时为 None，稍后由 RetrievalIndex 统一计算。

### 3.4.3 _split_long_paragraph

单段超过 max_tokens 时，按词切分，并保留 overlap。

```python
def _split_long_paragraph(paragraph, doc_id, metadata, max_tokens, overlap, heading, document_title):
    words = paragraph.split()
    step = max(1, max_tokens - overlap)
    chunks = []
    index = 0
    while index < len(words):
        end = min(len(words), index + max_tokens)
        text = f"{document_title}\n{heading}\n\n" + " ".join(words[index:end])
        chunks.append(_make_chunk(doc_id, text, metadata))
        if end == len(words):
            break
        index += step
    return chunks
```

这里有一个重要细节：每个 chunk 都把 document_title 与 heading 放到正文最前面。导入的岗位文档正文多为中文，公司名主要出现在标题中；如果不把标题放进 chunk 文本，检索 "Tencent" 时无法命中 Tencent 文档。这个修复直接影响真实语料检索质量。

### 3.4.4 _split_paragraphs

_split_paragraphs 负责组合段落并控制 token 上限。逻辑分三种情况：

- 段落本身超过 max_tokens：先把已有 current 输出，再调用 _split_long_paragraph。
- 加入当前段落会超过 max_tokens：输出当前 chunk，并从 current 尾部保留 overlap token。
- 正常情况：把段落加入 current。

```python
if para_tokens > max_tokens:
    if current:
        chunks.append(_make_chunk(doc_id, f"{document_title}\n{heading}\n\n" + "\n\n".join(current), metadata))
        current = []
    chunks.extend(_split_long_paragraph(...))
    continue
if current_tokens + para_tokens > max_tokens and current:
    chunks.append(_make_chunk(...))
    keep = 0
    kept = 0
    while keep < len(current) and kept < overlap:
        kept += count_tokens(current[keep])
        keep += 1
    current = current[-keep:] if keep else []
    current_tokens = kept
current.append(paragraph)
current_tokens += para_tokens
```

### 3.4.5 chunk_markdown_document

chunk_markdown_document 是整个切块入口。它先解析 front matter，再按行扫描标题。

```python
for line in body.splitlines():
    if line.startswith("#"):
        if current_paragraphs:
            sections.append((current_heading, current_paragraphs))
        current_heading = line.lstrip("#").strip()
        current_paragraphs = []
    elif line.strip():
        current_paragraphs.append(line.strip())
```

扫描结束后，如果 sections 为空但存在 current_heading，则生成一个只包含标题的 section。这一步保证只有标题、没有正文的文档也能被索引。最后对每个 section 调用 _split_paragraphs，返回 (metadata, chunks)。

## 3.5 BM25 实现

BM25Okapi 位于 core/retrieval/bm25.py。构造函数参数 k1=1.5、b=0.75。

```python
self.doc_freqs = [Counter(doc) for doc in corpus]
self.avgdl = sum(len(doc) for doc in corpus) / max(self.doc_count, 1)
df = Counter()
for doc in self.doc_freqs:
    df.update(doc.keys())
for term, freq in df.items():
    self.idf[term] = math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
```

单文档得分公式：

```text
score(q, d) = sum over term in q:
    idf(term) * (tf * (k1 + 1)) /
    (tf + k1 * (1 - b + b * len(d) / avgdl))
```

score 只对 query 中的 term 去重后计算；scores 返回全语料得分列表。BM25 的优点是不需要模型、可解释、对实体名和精确关键词很敏感。

## 3.6 向量模型

### 3.6.1 Embedder 协议

```python
class Embedder(Protocol):
    dimension: int
    def encode(self, texts: list[str]) -> list[list[float]]: ...
```

### 3.6.2 HashEmbedder

HashEmbedder 是离线默认实现，维度 256。它对每个 token 做 MD5，取前 4 字节决定维度，取第 5 字节决定正负号。

```python
digest = hashlib.md5(token.encode("utf-8")).digest()
index = int.from_bytes(digest[:4], "little") % self.dimension
sign = 1.0 if digest[4] % 2 == 0 else -1.0
vector[index] += sign
```

最后做 L2 归一化，因此余弦相似度等于点积。HashEmbedder 不是语义模型，但完全确定性、零依赖、零网络，适合 CI 和离线评测。

### 3.6.3 SentenceTransformerEmbedder

SentenceTransformerEmbedder 使用 sentence-transformers 加载模型，并按 device 参数决定 CPU 或 CUDA。encode 时使用 normalize_embeddings=True。

当前 BGE-M3 在没有安装 FlagEmbedding 时会通过这个类加载，因为 make_embedder 会在 FlagEmbeddingEmbedder 抛 RuntimeError 时回退到 SentenceTransformerEmbedder。

### 3.6.4 FlagEmbeddingEmbedder

FlagEmbeddingEmbedder 调用 BGEM3FlagModel。use_fp16 由 device 是否为 cuda 决定，dimension 固定为 1024。encode 使用 return_dense=True、max_length=8192。

## 3.7 重排层

重排层位于 core/retrieval/rerank.py，输入 RankedChunk 列表，输出重排后的列表。

IdentityReranker 直接返回前 top_k，用于离线与测试。

CrossEncoderReranker 使用 sentence-transformers.CrossEncoder。它把 (query, chunk.text) 组成 pairs，调用 model.predict，再按分数降序排序。

FlagReranker 使用 FlagEmbedding.FlagReranker。它调用 compute_score(pairs, normalize=True)，兼容单个 float 返回值。

make_reranker 根据 kind 选择实现：

- identity：IdentityReranker
- cross-encoder：CrossEncoderReranker
- bge-reranker 或 flag：FlagReranker

## 3.8 RetrievalIndex

RetrievalIndex 位于 core/retrieval/index.py，是检索层的门面。

### 3.8.1 构造

```python
def __init__(self, chunks, embedder=None, alpha=0.6):
    self.chunks = list(chunks)
    self.embedder = embedder or HashEmbedder()
    self.alpha = alpha
    self.bm25 = BM25Okapi([chunk.tokens for chunk in self.chunks])
    if self.embedder is not None and all(chunk.embedding is None for chunk in self.chunks):
        embeddings = self.embedder.encode([chunk.text for chunk in self.chunks])
        for chunk, vector in zip(self.chunks, embeddings):
            chunk.embedding = vector
```

如果 chunks 已经有 embedding，构造函数不会重复计算。这使 from_json 读取缓存后可以直接搜索。

### 3.8.2 from_corpus

from_corpus 遍历 corpus_dir 中所有 Markdown，调用 chunk_markdown_document，汇总 chunks，再交给构造函数。

### 3.8.3 缓存

to_json 把 alpha 和 chunks 写入 JSON，每个 chunk 包含 id、doc_id、text、metadata、tokens、embedding。

from_json 读取 JSON 并重建 Chunk。load_or_build 先尝试读取缓存，失败或不存在时再从 corpus 构建，并写回缓存。

GPU 路径的核心命令是：

```powershell
python scripts\build_embedding_index.py --mode bge-m3 --model BAAI/bge-m3 --device cuda --output data\storage\index-bge-m3.json
```

### 3.8.4 混合打分

```python
query_tokens = tokenize(query)
bm25_scores = self.bm25.scores(query_tokens)
query_embedding = self.embedder.encode([query])[0]
dense_scores = [self._cosine(query_embedding, chunk.embedding) if chunk.embedding else 0.0 for chunk in self.chunks]
combined = [effective_alpha * bm + (1 - effective_alpha) * dense for bm, dense in zip(bm25_scores, dense_scores)]
```

alpha 默认 0.6，表示 BM25 权重 0.6，dense 权重 0.4。排序后先取 max(top_k * 2, top_k) 个候选，再交给 reranker。这样做是为了让重排器有更大的候选池，而不是只对最终 top_k 重排。

### 3.8.5 to_sources

to_sources 把 RankedChunk 转成 Source。title 取 metadata["title"]，url 取 metadata["source_url"]，snippet 取 chunk.text 前 500 字符，metadata 中保留 doc_id、heading、score。

## 3.9 语料来源

### 3.9.1 种子语料 scripts/seed_corpus.py

seed_corpus.py 内置 7 份文档，覆盖 Shopee、TikTok、Grab、NVIDIA、GovTech/A*STAR、2026 招聘趋势，以及一份用于矛盾薪资演示的 Shopee stipend survey。运行后写入 data/corpus/seed/。

### 3.9.2 真实岗位清单 scripts/import_sg_jobs.py

import_sg_jobs.py 读取 新加坡AI实习机会清单.docx 的第一张表，按表头映射：

```python
HEADER_MAP = {
    "公司 / 项目": "company",
    "岗位方向": "roles",
    "薪资待遇": "compensation",
    "能否转正": "return_offer",
    "笔试 / 面试": "hiring_process",
    "官方申请链接": "apply_url",
    "备注 / 推荐理由": "notes",
}
```

脚本为每行生成一个 Markdown 文件，并生成 00_master-list.md 汇总表。当前导入 38 条记录，覆盖 ByteDance/TikTok、Shopee、Alibaba/Lazada、Tencent、Huawei、Sea/Garena、Grab、Google、Microsoft、Meta、Amazon、NVIDIA、Apple、SAP、PayPal、Visa、Salesforce、DBS、OCBC、UOB、Standard Chartered、JPMorgan、GIC、GovTech、A*STAR、ST Engineering、Singtel、Razer、Micron、Cynapse、Guidesify、AIPilot、ESGPedia、X Star、Hong Ye、YY Circle、LinkWave 等。

导入后的文档结构固定为 Roles、Compensation、Return Offer、Hiring Process、Notes 五段，因此每个岗位可以被切块、检索、合并，并进入最终报告。
