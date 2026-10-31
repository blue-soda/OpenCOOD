// Exact first-overflow search. Every prefix is checked; no monotonicity assumption.
#include <algorithm>
#include <cstdint>
#include <vector>
#include <zlib.h>
#include <omp.h>

extern "C" const char* snapshot_zlib_version() { return zlibVersion(); }

static uint64_t packet_size(const std::vector<uint8_t>& rows,
                            const uint32_t* coords, int64_t nr,
                            uint64_t overhead) {
    std::vector<uint8_t> raw, masks, values;
    raw.reserve(nr * 8); masks.reserve(nr * 4); values.reserve(nr * 8);
    unsigned pending = 0, nibble = 0;
    for (int64_t r = 0; r < nr; ++r) {
        uint32_t mask = 0;
        for (int c = 0; c < 32; ++c) if (rows[r * 32 + c]) mask |= uint32_t(1) << c;
        if (!mask) continue;
        for (int b = 0; b < 4; ++b) {
            raw.push_back(uint8_t(coords[r] >> (8 * b)));
            masks.push_back(uint8_t(mask >> (8 * b)));
        }
        for (int c = 0; c < 32; ++c) {
            unsigned v = rows[r * 32 + c];
            if (!v) continue;
            if (!pending) { nibble = v; pending = 1; }
            else { values.push_back(uint8_t(nibble | (v << 4))); pending = 0; }
        }
    }
    if (raw.empty()) return 0;
    if (pending) values.push_back(uint8_t(nibble));
    raw.insert(raw.end(), masks.begin(), masks.end());
    raw.insert(raw.end(), values.begin(), values.end());
    uLongf length = compressBound(raw.size());
    std::vector<uint8_t> compressed(length);
    if (compress2(compressed.data(), &length, raw.data(), raw.size(), Z_DEFAULT_COMPRESSION) != Z_OK)
        return UINT64_MAX;
    return overhead + length;
}

extern "C" int64_t snapshot_first_overflow(const uint32_t* coords, int64_t nr,
        const int64_t* slots, const int64_t* channels, int64_t n,
        uint64_t overhead, uint64_t budget, int threads, int chunk,
        int64_t* rejected_bytes) {
    if (nr < 0 || n < 0 || threads < 1 || threads > 32 || chunk < threads) return -1;
    std::vector<uint8_t> base(nr * 32, 0);
    *rejected_bytes = -1;
    for (int64_t begin = 0; begin < n; begin += chunk) {
        int64_t end = std::min(n, begin + chunk);
        std::vector<int64_t> first(threads, end), rejected(threads, -1);
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (int worker = 0; worker < threads; ++worker) {
            int64_t lo = begin + (end - begin) * worker / threads;
            int64_t hi = begin + (end - begin) * (worker + 1) / threads;
            if (lo == hi) continue;
            std::vector<uint8_t> rows(base);
            for (int64_t j = begin; j < lo; ++j) ++rows[slots[j] * 32 + channels[j]];
            for (int64_t j = lo; j < hi; ++j) {
                ++rows[slots[j] * 32 + channels[j]];
                uint64_t size = packet_size(rows, coords, nr, overhead);
                if (size > budget) {
                    first[worker] = j;
                    rejected[worker] = size == UINT64_MAX ? -2 : int64_t(size);
                    break;
                }
            }
        }
        auto it = std::min_element(first.begin(), first.end());
        if (*it < end) {
            *rejected_bytes = rejected[it - first.begin()];
            return *rejected_bytes == -2 ? -1 : *it;
        }
        for (int64_t j = begin; j < end; ++j) ++base[slots[j] * 32 + channels[j]];
    }
    return n;
}
