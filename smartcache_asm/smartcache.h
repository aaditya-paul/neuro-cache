/**
 * SMARTCACHE AI — High-Performance Native Assembly Engine
 * Header Specification & C ABI Definitions
 * 
 * Hardware-level, cache-line aligned data structures for nanosecond-latency
 * cache replacement decisions via x86-64 SIMD and register-optimized state.
 */

#ifndef SMARTCACHE_H
#define SMARTCACHE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SMARTCACHE_MAX_CAPACITY 1024
#define SMARTCACHE_MAX_GHOST    1024
#define SMARTCACHE_WINDOW_MAX   256
#define SMARTCACHE_HASH_SIZE    2048
#define SMARTCACHE_HASH_MASK    (SMARTCACHE_HASH_SIZE - 1)

#pragma pack(push, 8)

/**
 * Cache slot structure (48 bytes, aligned).
 */
typedef struct {
    int64_t  key;               // Item identifier (0 if unused, or check is_valid)
    int64_t  last_access_step;  // Simulation step of most recent access
    int64_t  freq;              // Total historical access count
    int32_t  prev_lru;          // Index of previous slot in LRU recency list (-1 if head)
    int32_t  next_lru;          // Index of next slot in LRU recency list (-1 if tail)
    uint32_t is_valid;          // 1 if slot is actively holding a cached block, 0 otherwise
    uint32_t pad[3];            // Padding for structure alignment
} SmartCacheSlot;

/**
 * Ghost cache entry for credit/regret attribution.
 */
typedef struct {
    int64_t  key;               // Evicted item identifier
    int64_t  evict_step;        // Simulation step when eviction occurred
    uint32_t is_valid;          // 1 if valid, 0 if empty
    uint32_t pad;
} SmartGhostSlot;

/**
 * Core SmartCache engine state.
 */
typedef struct {
    // Configuration parameters
    int32_t capacity;           // Cache capacity C
    int32_t ghost_capacity;     // Dual shadow buffer capacity
    int32_t ghost_ttl;          // Temporal expiration horizon
    int32_t deterministic;      // 1 for deterministic utility choice, 0 for stochastic
    
    // Hyperparameters & thresholds
    double  learning_rate;      // Base learning rate lambda
    double  discount;           // Temporal discount factor gamma (default 1.0)
    double  damping_factor;     // Single-miss damping multiplier (0.35)
    double  shift_threshold;    // Hit-rate drop threshold (0.20)
    double  epsilon_min;        // Minimum exploration floor (0.005)
    double  halflife_mult;      // Halflife multiplier (5.0)
    double  freq_cap;           // Maximum frequency saturation cap (15.0)
    double  baseline_hit_rate;  // Rolling EMA baseline hit rate
    
    // Expert weights (sum to 1.0)
    double  w_lru;              // Weight assigned to LRU expert
    double  w_lfu;              // Weight assigned to LFU expert
    
    // Step and telemetry counters
    int64_t current_step;       // Global request counter
    int64_t hits;               // Total cache hits
    int64_t misses;             // Total cache misses
    int64_t evictions;          // Total evictions
    int64_t ghost_hits_lru;     // Total LRU ghost hits (regrets)
    int64_t ghost_hits_lfu;     // Total LFU ghost hits (regrets)
    int64_t expired_ghost_entries; // Expired ghost entries ignored by TTL
    
    // Cache list and state indices
    int32_t current_size;       // Number of items currently resident in cache
    int32_t lru_head;           // Slot index of Least Recently Used item
    int32_t lru_tail;           // Slot index of Most Recently Used item
    int32_t last_chosen_expert; // 0=NONE, 1=LRU, 2=LFU, 3=CONSENSUS
    int32_t is_shift_active;    // 1 if workload shift is detected, 0 otherwise
    
    // Fast pseudo-random number generator state (XorShift64)
    uint64_t rng_state;
    
    // Rolling outcome window for workload shift detection
    int32_t window_size;
    int32_t window_head;
    int32_t window_count;
    int32_t window_hit_count;   // O(1) running accumulator of hit outcomes in window
    uint8_t window_hits[SMARTCACHE_WINDOW_MAX];
    int64_t window_keys[SMARTCACHE_WINDOW_MAX];
    
    // Recent ghost hits tracking for regret momentum & damping
    int32_t recent_ghost_head;
    int32_t recent_ghost_count;
    int32_t recent_ghost_experts[SMARTCACHE_WINDOW_MAX]; // 1=LRU, 2=LFU
    int64_t recent_ghost_steps[SMARTCACHE_WINDOW_MAX];
    
    // Fast hash table for O(1) slot lookup by key
    // Stores (slot_index + 1), or 0 if empty
    int32_t hash_table[SMARTCACHE_HASH_SIZE];
    
    // Active cache slots array
    SmartCacheSlot slots[SMARTCACHE_MAX_CAPACITY];
    
    // Dual shadow ghost buffers (circular FIFO ring buffers)
    int32_t ghost_lru_count;
    int32_t ghost_lru_head;
    SmartGhostSlot ghost_lru[SMARTCACHE_MAX_GHOST];
    
    int32_t ghost_lfu_count;
    int32_t ghost_lfu_head;
    SmartGhostSlot ghost_lfu[SMARTCACHE_MAX_GHOST];
} SmartCacheState;

#pragma pack(pop)

/**
 * Public Assembly Exports (x86-64 Fastcall / Microsoft ABI)
 */

void smartcache_init(
    SmartCacheState* state,
    int32_t capacity,
    double learning_rate,
    double damping_factor,
    double shift_threshold,
    double halflife_mult,
    double freq_cap,
    uint64_t seed
);

void smartcache_reset(SmartCacheState* state);

int32_t smartcache_access(
    SmartCacheState* state,
    int64_t key,
    int32_t* out_is_hit,
    int64_t* out_evicted_key,
    int32_t* out_expert
);

int64_t smartcache_choose_victim(
    SmartCacheState* state,
    int32_t* out_victim_slot,
    int32_t* out_chosen_expert
);

void smartcache_get_weights(SmartCacheState* state, double* out_w_lru, double* out_w_lfu);

void smartcache_run_trace(
    SmartCacheState* state,
    const int64_t* trace,
    int64_t trace_len,
    int64_t* out_hits,
    int64_t* out_misses,
    int64_t* out_evictions,
    double*  out_avg_decision_ns
);

#ifdef __cplusplus
}
#endif

#endif // SMARTCACHE_H
