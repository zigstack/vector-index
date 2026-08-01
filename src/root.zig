//! vector-index: a pure-Zig vector similarity index with an optional FAISS
//! accelerator.
//!
//! The flat exact scan is the implementation of record: no native
//! dependencies, cross-compiles statically everywhere Zig does, and serves
//! as the reference any accelerator must agree with. When `libfaiss_c` is
//! present at runtime (dlopen -- never a build or install requirement), a
//! FAISS flat inner-product index mirrors the vectors and serves searches.
//!
//! Vectors are expected L2-normalized so inner product equals cosine
//! similarity; the index neither checks nor normalizes. Indices returned by
//! `search` are insertion positions. Not thread-safe: callers synchronize.
//!
//! Extracted from the loom project's RAG store; HNSW is the planned second
//! index type for ANN-scale collections.
const std = @import("std");

pub const faiss = @import("faiss.zig");

pub const Hit = struct { idx: usize, score: f32 };

pub const Index = struct {
    gpa: std.mem.Allocator,
    dim: usize,
    /// Contiguous row-major vector storage -- the scan's cache locality.
    data: std.ArrayListUnmanaged(f32) = .empty,
    accel: ?faiss.Index,

    pub fn init(gpa: std.mem.Allocator, dim: usize) Index {
        return .{ .gpa = gpa, .dim = dim, .accel = faiss.Index.init(dim) };
    }

    pub fn deinit(self: *Index) void {
        self.data.deinit(self.gpa);
    }

    pub fn count(self: *const Index) usize {
        return self.data.items.len / self.dim;
    }

    /// Copy one vector in. The accelerator mirrors it; a mirror failure is
    /// tolerated (search falls back to the scan while counts disagree).
    pub fn add(self: *Index, vec: []const f32) !void {
        std.debug.assert(vec.len == self.dim);
        try self.data.appendSlice(self.gpa, vec);
        if (self.accel) |*a| _ = a.add(vec);
    }

    /// Top-k by inner product, best first, into `out`; returns the hit
    /// count. Uses FAISS only while its mirror count matches -- the exact
    /// scan is always correct and always available.
    pub fn search(self: *Index, query: []const f32, k: usize, out: []Hit) usize {
        std.debug.assert(query.len == self.dim);
        const n_vecs = self.count();
        if (n_vecs == 0 or k == 0 or out.len == 0) return 0;
        const kk = @min(k, @min(n_vecs, out.len));
        if (self.accel) |*a| {
            if (a.count() == n_vecs and kk <= 16) {
                var scores: [16]f32 = undefined;
                var ids: [16]i64 = undefined;
                const n = a.search(query, kk, scores[0..kk], ids[0..kk]);
                if (n > 0) {
                    for (0..n) |i| out[i] = .{ .idx = @intCast(ids[i]), .score = scores[i] };
                    return n;
                }
            }
        }
        return self.scan(query, kk, out);
    }

    /// The exact reference scan: insertion-sorted top-k inner products.
    pub fn scan(self: *const Index, query: []const f32, kk: usize, out: []Hit) usize {
        var n: usize = 0;
        var idx: usize = 0;
        var off: usize = 0;
        while (off < self.data.items.len) : ({
            off += self.dim;
            idx += 1;
        }) {
            const row = self.data.items[off..][0..self.dim];
            var s: f32 = 0;
            for (row, query) |a, b| s += a * b;
            var pos = n;
            while (pos > 0 and out[pos - 1].score < s) pos -= 1;
            if (pos >= kk) continue;
            if (n < kk) n += 1;
            var j = n - 1;
            while (j > pos) : (j -= 1) out[j] = out[j - 1];
            out[pos] = .{ .idx = idx, .score = s };
        }
        return n;
    }
};

fn unit(comptime dim: usize, seed: u64) [dim]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var v: [dim]f32 = undefined;
    var norm: f32 = 0;
    for (&v) |*x| {
        x.* = rnd.float(f32) - 0.5;
        norm += x.* * x.*;
    }
    const inv = 1.0 / @sqrt(norm);
    for (&v) |*x| x.* *= inv;
    return v;
}

test "exact self-match ranks first with score ~1" {
    const gpa = std.testing.allocator;
    var ix = Index.init(gpa, 32);
    defer ix.deinit();
    const a = unit(32, 1);
    const b = unit(32, 2);
    const c = unit(32, 3);
    try ix.add(&a);
    try ix.add(&b);
    try ix.add(&c);
    var hits: [3]Hit = undefined;
    const n = ix.search(&b, 3, &hits);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqual(@as(usize, 1), hits[0].idx);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), hits[0].score, 1e-5);
    try std.testing.expect(hits[0].score >= hits[1].score and hits[1].score >= hits[2].score);
}

test "accelerator, when present, agrees with the reference scan" {
    const gpa = std.testing.allocator;
    var ix = Index.init(gpa, 24);
    defer ix.deinit();
    for (0..40) |i| {
        const v = unit(24, 100 + i);
        try ix.add(&v);
    }
    const q = unit(24, 7);
    var got: [5]Hit = undefined;
    var ref: [5]Hit = undefined;
    const ng = ix.search(&q, 5, &got);
    const nr = ix.scan(&q, 5, &ref);
    try std.testing.expectEqual(nr, ng);
    for (got[0..ng], ref[0..nr]) |g, r| {
        try std.testing.expectEqual(r.idx, g.idx);
        try std.testing.expectApproxEqAbs(r.score, g.score, 1e-5);
    }
}

test "k larger than the collection clamps" {
    const gpa = std.testing.allocator;
    var ix = Index.init(gpa, 8);
    defer ix.deinit();
    const v = unit(8, 9);
    try ix.add(&v);
    var hits: [4]Hit = undefined;
    try std.testing.expectEqual(@as(usize, 1), ix.search(&v, 4, &hits));
}
