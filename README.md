# vector-index

A pure-Zig vector similarity index with an optional FAISS accelerator.

- **Flat exact search** (inner product; pass L2-normalized vectors for
  cosine) is the implementation of record: no native dependencies, and it
  cross-compiles statically everywhere Zig does.
- **FAISS**, when `libfaiss_c` is present at runtime, mirrors the vectors in
  a flat IP index and serves searches (dlopen — never a build or install
  requirement). The exact scan is the reference the accelerator must agree
  with, and the test suite asserts it.
- **HNSW** in pure Zig is the planned second index type for ANN-scale
  collections.

Not thread-safe by design: callers hold their own lock. `search` returns
insertion indices, best first.

```zig
const vi = @import("vector_index");
var ix = vi.Index.init(gpa, 384);
defer ix.deinit();
try ix.add(&embedding);           // L2-normalized []f32
var hits: [5]vi.Hit = undefined;
const n = ix.search(&query, 5, &hits);
```

```sh
zig build test
```

Extracted from [loom](https://github.com/ch4r10t33r/loom), where it serves
the gossiped RAG chunk store.

## License

MIT
