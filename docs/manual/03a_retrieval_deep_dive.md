# 第三章补充 检索层逐函数走读

## 3A.1 tokenize 的完整行为

```python
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]
```

正则有两个分支，用 | 连接。第一个分支匹配一个或多个 ASCII 字母、数字、下划线；第二个分支匹配一个或多个汉字。findall 按出现顺序返回所有匹配，最后统一转小写。

@CAPTION: 表 3A-1 tokenize 示例
:::table
输入|输出|说明
Shopee AI|["shopee", "ai"]|英文按词
新加坡 AI|["新加坡", "ai"]|中文连续串作为一个 token
BGE-M3|["bge", "m3"]|连字符不作为 token 一部分
S$2,500-4,000|["s", "2", "500", "4", "000"]|货币符号与逗号被切掉
中文混合abc|["中文混合", "abc"]|中文与英文分段
:::

## 3A.2 parse_front_matter 的控制流

parse_front_matter 有两个几乎等价的分支，原因是要兼容开头就是 --- 的情况。核心步骤：

1. 如果 markdown 不以 --- 开头，返回空 metadata 和原文。
2. 把文本按行切分。
3. 找到第二个 --- 的行号 end。
4. 解析 lines[1:end] 中的 key: value。
5. 如果 value 形如 [a, b]，转成 list。
6. 返回 metadata 与 lines[end+1:] 拼接的 body。

这个解析器不支持嵌套 YAML、引号中的冒号、多行列表。当前语料的 front matter 都是简单字段，所以足够。

## 3A.3 _chunk_id 为什么使用 SHA-1 而不是 uuid

```python
digest = hashlib.sha1(f"{doc_id}:{text}".encode()).hexdigest()[:12]
return f"{doc_id}-{digest}"
```

使用内容哈希有三个好处：

- 同一文档同一内容重复构建时 ID 稳定。
- 不同内容不会复用同一个 ID。
- 不需要维护全局计数器，便于多进程构建。

截断到 12 位十六进制后碰撞概率很低，适合当前 200 多个 chunk 的规模。

## 3A.4 _split_long_paragraph 的边界

```python
step = max(1, max_tokens - overlap)
```

如果 max_tokens=80、overlap=20，step=60。每次窗口向前移动 60 个词，窗口长度 80，因此相邻 chunk 有 20 个词重叠。如果 overlap >= max_tokens，step 至少为 1，避免死循环。

每个窗口的文本前面都会加 document_title 和 heading。这个策略增加 token 数，但换来实体名与标题信息的稳定保留。

## 3A.5 _split_paragraphs 的三个分支

分支一：段落本身超长。

```python
if para_tokens > max_tokens:
    if current:
        chunks.append(_make_chunk(...))
        current = []
    chunks.extend(_split_long_paragraph(...))
    continue
```

先把已经积累的 current 作为一个 chunk 输出，再单独处理超长段落。这样不会把超长段落和普通段落混在同一个 chunk。

分支二：加入段落会超限。

```python
if current_tokens + para_tokens > max_tokens and current:
    chunks.append(_make_chunk(...))
    keep = 0
    kept = 0
    while keep < len(current) and kept < overlap:
        kept += count_tokens(current[keep])
        keep += 1
    current = current[-keep:] if keep else []
    current_tokens = kept
```

它会从 current 的尾部保留若干段落，累计 token 接近 overlap。注意保留的是整段，不是切开的句子。

分支三：正常累加。

```python
current.append(paragraph)
current_tokens += para_tokens
```

循环结束后，如果 current 非空，再输出最后一个 chunk。

## 3A.6 chunk_markdown_document 的标题处理

扫描 body 时，任何以 # 开头的行都被当成标题。

```python
if line.startswith("#"):
    if current_paragraphs:
        sections.append((current_heading, current_paragraphs))
    current_heading = line.lstrip("#").strip()
    current_paragraphs = []
```

它不区分 #、##、### 的层级，只把标题文本当作 heading。这样实现简单，但会丢失层级的父子关系。当前语料标题层级不深，因此可接受。

如果文档只有标题没有正文：

```python
if not sections and current_heading:
    sections.append((current_heading, []))
```

随后 _split_paragraphs 对空 paragraphs 返回一个包含标题的 chunk。这保证纯标题文档不会被索引漏掉。

## 3A.7 BM25 的时间复杂度

构造阶段：

- 遍历 corpus 计算 doc_freqs 与 df：O(N * L)，N 为文档数，L 为平均 token 数。
- 计算 idf：O(V)，V 为词表大小。

查询阶段：

- score 对 set(query_tokens) 遍历：O(Q)。
- 每个 term 做一次 Counter 查询：O(1)。
- scores 对每个文档调用 score：O(N * Q)。

在当前 200 多个 chunk、查询十几个 token 的规模下，BM25 开销可以忽略。

## 3A.8 HashEmbedder 的碰撞问题

HashEmbedder 把 token 映射到 256 维。不同 token 可能落到同一维度，发生碰撞。使用 sign 正负后，部分碰撞会相互抵消，部分会增加。它不是语义模型，但对共享关键词的查询仍然有效。

这种设计的定位是 CI 与离线 demo，不是生产检索。真实语义检索必须使用 SentenceTransformerEmbedder 或 FlagEmbeddingEmbedder。

## 3A.9 SentenceTransformerEmbedder 与 FlagEmbeddingEmbedder 的差异

@CAPTION: 表 3A-2 两种向量实现差异
:::table
维度|SentenceTransformerEmbedder|FlagEmbeddingEmbedder
依赖|sentence-transformers|FlagEmbedding
模型接口|SentenceTransformer.encode|BGEM3FlagModel.encode
归一化|normalize_embeddings=True|由模型输出 dense_vecs
维度|从模型读取|固定 1024
设备|device 参数|use_fp16 由 cuda 决定
回退|无|make_embedder 在 FlagEmbedding 缺失时回退到 SentenceTransformer
:::

## 3A.10 CrossEncoderReranker 的输入输出

```python
pairs = [(query, item.chunk.text) for item in items]
scores = self.model.predict(pairs)
scored = sorted(
    (RankedChunk(item.chunk, float(score)) for item, score in zip(items, scores)),
    key=lambda x: x.score,
    reverse=True,
)
return scored[:top_k]
```

输入是候选 RankedChunk 列表，输出是重排后的 RankedChunk 列表。CrossEncoder 的分数不是概率，只是排序信号。当前没有做分数阈值过滤。

## 3A.11 RetrievalIndex.search 的完整计算顺序

1. 如果 chunks 为空，直接返回 []。
2. 对 query 分词。
3. 计算 BM25 分数列表。
4. 计算 query embedding。
5. 对每个 chunk 计算 cosine 分数。
6. 用 alpha 组合 BM25 与 dense 分数。
7. 按组合分数降序排序。
8. 取 max(top_k * 2, top_k) 个候选。
9. 调用 make_reranker 得到重排器。
10. 重排器返回最终 top_k。

## 3A.12 缓存 JSON 的结构

```json
{
  "alpha": 0.6,
  "chunks": [
    {
      "id": "doc-abc123",
      "doc_id": "doc",
      "text": "...",
      "metadata": {...},
      "tokens": ["..."],
      "embedding": [0.1, 0.2, ...]
    }
  ]
}
```

缓存包含 embedding，因此加载后不需要重新编码 corpus。query 仍然需要 embedder.encode([query])，所以 from_json 必须传入 embedder。

## 3A.13 检索层的边界

第一，当前 BM25 与 dense 分数直接线性加权，没有做分数归一化。BM25 分数范围与 cosine 分数范围不同，alpha=0.6 是经验值，不是理论最优。

第二，实体过滤使用 title/snippet 的字符串包含，不是 metadata 的精确 entity 字段。

第三，检索没有去重同一个 URL 的重复内容，只在 _merge_by_doc 中按 doc_id 合并。

第四，缓存没有版本号。如果 chunker 或 embedding 模型改变，旧缓存仍会被加载。应增加 cache version 或 model hash。

第五，BGE-M3 与 reranker 的 GPU 驻留没有统一管理，多个模型同时加载时可能超过 8 GB 显存。
