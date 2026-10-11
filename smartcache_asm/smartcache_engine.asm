; ==============================================================================
; SMARTCACHE AI — High-Performance Native x86-64 Assembly Engine
; 
; Architecture: x86-64 Microsoft ABI (Fastcall)
; Hand-crafted assembly implementing online LeCaR adaptive caching, dual ghost
; credit attribution, decayed effective utility evaluation, workload shift detection,
; and bounded Exponential Gradient updates at hardware-level nanosecond latency.
; ==============================================================================

.data
    ALIGN 16
    c_ln2           REAL8 0.6931471805599453094172321
    c_log2e         REAL8 1.4426950408889634073599247
    c_half          REAL8 0.5
    c_one           REAL8 1.0
    c_zero          REAL8 0.0
    c_two           REAL8 2.0
    c_p1            REAL8 0.6931471805599453
    c_p2            REAL8 0.2402265069591007
    c_p3            REAL8 0.0555041086648215
    c_p4            REAL8 0.0096181291076284
    c_exp_min       REAL8 -700.0
    c_ema_alpha     REAL8 0.05
    c_ema_beta      REAL8 0.95
    c_logistic_k    REAL8 3.0
    c_clamp_min     REAL8 -8.0
    c_clamp_max     REAL8 8.0

.code

; Structure offset constants
OFFSET_CAPACITY              EQU 0
OFFSET_GHOST_CAPACITY        EQU 4
OFFSET_GHOST_TTL             EQU 8
OFFSET_DETERMINISTIC         EQU 12
OFFSET_LEARNING_RATE         EQU 16
OFFSET_DISCOUNT              EQU 24
OFFSET_DAMPING_FACTOR        EQU 32
OFFSET_SHIFT_THRESHOLD       EQU 40
OFFSET_EPSILON_MIN           EQU 48
OFFSET_HALFLIFE_MULT         EQU 56
OFFSET_FREQ_CAP              EQU 64
OFFSET_BASELINE_HIT_RATE     EQU 72
OFFSET_W_LRU                 EQU 80
OFFSET_W_LFU                 EQU 88
OFFSET_CURRENT_STEP          EQU 96
OFFSET_HITS                  EQU 104
OFFSET_MISSES                EQU 112
OFFSET_EVICTIONS             EQU 120
OFFSET_GHOST_HITS_LRU        EQU 128
OFFSET_GHOST_HITS_LFU        EQU 136
OFFSET_EXPIRED_GHOST         EQU 144
OFFSET_CURRENT_SIZE          EQU 152
OFFSET_LRU_HEAD              EQU 156
OFFSET_LRU_TAIL              EQU 160
OFFSET_LAST_EXPERT           EQU 164
OFFSET_IS_SHIFT_ACTIVE       EQU 168
OFFSET_RNG_STATE             EQU 176
OFFSET_WINDOW_SIZE           EQU 184
OFFSET_WINDOW_HEAD           EQU 188
OFFSET_WINDOW_COUNT          EQU 192
OFFSET_WINDOW_HIT_COUNT      EQU 196
OFFSET_WINDOW_HITS           EQU 200
OFFSET_WINDOW_KEYS           EQU 456
OFFSET_RECENT_GHOST_HEAD     EQU 2504
OFFSET_RECENT_GHOST_COUNT    EQU 2508
OFFSET_RECENT_GHOST_EXP      EQU 2512
OFFSET_RECENT_GHOST_STEPS    EQU 3536
OFFSET_HASH_TABLE            EQU 5584
OFFSET_SLOTS                 EQU 13776
OFFSET_GHOST_LRU_COUNT       EQU 62928
OFFSET_GHOST_LRU_HEAD        EQU 62932
OFFSET_GHOST_LRU             EQU 62936
OFFSET_GHOST_LFU_COUNT       EQU 87512
OFFSET_GHOST_LFU_HEAD        EQU 87516
OFFSET_GHOST_LFU             EQU 87520

SLOT_SIZE                    EQU 48
SLOT_KEY                     EQU 0
SLOT_LAST_ACCESS             EQU 8
SLOT_FREQ                    EQU 16
SLOT_PREV_LRU                EQU 24
SLOT_NEXT_LRU                EQU 28
SLOT_IS_VALID                EQU 32

GHOST_SLOT_SIZE              EQU 24
GHOST_KEY                    EQU 0
GHOST_EVICT_STEP             EQU 8
GHOST_IS_VALID               EQU 16

STATE_TOTAL_SIZE             EQU 112096

; ==============================================================================
; Helper: asm_fast_exp
; Input:  XMM0 = double x (x <= 0.0)
; Output: XMM0 = exp(x)
; Clobbers: RAX, RDX, XMM1, XMM2, XMM3
; ==============================================================================
asm_fast_exp PROC
    comisd xmm0, c_exp_min
    jb @exp_zero
    comisd xmm0, c_zero
    ja @exp_one

    ; t = x * log2e
    movsd xmm1, xmm0
    mulsd xmm1, c_log2e

    ; k = floor(t)
    roundsd xmm2, xmm1, 1 ; 1 = Round down (floor)
    cvtsd2si rax, xmm2    ; integer k

    ; f = t - k (in [0, 1))
    subsd xmm1, xmm2

    ; Polynomial approximation for 2^f:
    ; P(f) = 1.0 + f * (p1 + f * (p2 + f * (p3 + f * p4)))
    movsd xmm0, c_p4
    mulsd xmm0, xmm1
    addsd xmm0, c_p3
    mulsd xmm0, xmm1
    addsd xmm0, c_p2
    mulsd xmm0, xmm1
    addsd xmm0, c_p1
    mulsd xmm0, xmm1
    addsd xmm0, c_one

    ; Scale by 2^k via IEEE-754 exponent field
    add rax, 1023
    cmp rax, 0
    jle @exp_zero
    cmp rax, 2047
    jge @exp_overflow
    shl rax, 52
    movq xmm3, rax
    mulsd xmm0, xmm3
    ret

@exp_zero:
    xorpd xmm0, xmm0
    ret

@exp_one:
    movsd xmm0, c_one
    ret

@exp_overflow:
    movsd xmm0, c_one
    ret
asm_fast_exp ENDP


; ==============================================================================
; Helper: asm_xorshift64
; Input:  RCX = state pointer
; Output: XMM0 = uniform random double in [0.0, 1.0)
; Clobbers: RAX, RDX
; ==============================================================================
asm_xorshift64 PROC
    mov rax, qword ptr [rcx + OFFSET_RNG_STATE]
    test rax, rax
    jnz @rng_valid
    mov rax, 88172645463325252 ; default fallback seed
@rng_valid:
    mov rdx, rax
    shl rdx, 13
    xor rax, rdx
    mov rdx, rax
    shr rdx, 7
    xor rax, rdx
    mov rdx, rax
    shl rdx, 17
    xor rax, rdx
    mov qword ptr [rcx + OFFSET_RNG_STATE], rax

    ; Convert lower 52 bits to [1.0, 2.0) double, then subtract 1.0
    mov rdx, 000FFFFFFFFFFFFFh
    and rax, rdx
    mov rdx, 3FF0000000000000h
    or rax, rdx
    movq xmm0, rax
    subsd xmm0, c_one
    ret
asm_xorshift64 ENDP


; ==============================================================================
; smartcache_init
; C Prototype:
; void smartcache_init(
;     SmartCacheState* state,
;     int32_t capacity,
;     double learning_rate,
;     double damping_factor,
;     double shift_threshold,
;     double halflife_mult,
;     double freq_cap,
;     uint64_t seed
; );
; ==============================================================================
smartcache_init PROC
    push rdi
    push rsi
    push rbx

    test rcx, rcx
    jz @init_ret

    ; Zero-initialize state buffer (112096 bytes)
    mov rdi, rcx
    xor eax, eax
    mov ecx, STATE_TOTAL_SIZE / 8
    rep stosq
    mov rcx, rdi
    sub rcx, STATE_TOTAL_SIZE ; Restore rcx = state

    ; Set capacity
    cmp edx, 1
    jge @cap_ok
    mov edx, 1
@cap_ok:
    cmp edx, 1024
    jle @cap_bounded
    mov edx, 1024
@cap_bounded:
    mov dword ptr [rcx + OFFSET_CAPACITY], edx

    ; ghost_capacity = capacity
    mov dword ptr [rcx + OFFSET_GHOST_CAPACITY], edx

    ; ghost_ttl = max(10, capacity * 3)
    mov eax, edx
    imul eax, 3
    cmp eax, 10
    jge @ttl_ok
    mov eax, 10
@ttl_ok:
    mov dword ptr [rcx + OFFSET_GHOST_TTL], eax

    ; deterministic = 1
    mov dword ptr [rcx + OFFSET_DETERMINISTIC], 1

    ; discount = 1.0
    movsd xmm0, c_one
    movsd real8 ptr [rcx + OFFSET_DISCOUNT], xmm0

    ; epsilon_min = 0.005
    mov rax, 3F747AE147AE147Bh ; 0.005 double
    mov qword ptr [rcx + OFFSET_EPSILON_MIN], rax

    ; learning_rate
    comisd xmm2, c_zero
    ja @lr_passed
    ; Auto compute sqrt(ln(2)/C)
    cvtsi2sd xmm0, edx
    movsd xmm1, c_ln2
    divsd xmm1, xmm0
    sqrtsd xmm2, xmm1
@lr_passed:
    movsd real8 ptr [rcx + OFFSET_LEARNING_RATE], xmm2

    ; damping_factor
    comisd xmm3, c_zero
    ja @damp_passed
    mov rax, 3FD6666666666666h ; 0.35 double
    mov qword ptr [rcx + OFFSET_DAMPING_FACTOR], rax
    jmp @damp_done
@damp_passed:
    movsd real8 ptr [rcx + OFFSET_DAMPING_FACTOR], xmm3
@damp_done:

    ; Stack arguments:
    movsd xmm0, real8 ptr [rsp + 64] ; shift_threshold
    comisd xmm0, c_zero
    ja @shift_passed
    mov rax, 3FC999999999999Ah ; 0.20 double
    mov qword ptr [rcx + OFFSET_SHIFT_THRESHOLD], rax
    jmp @shift_done
@shift_passed:
    movsd real8 ptr [rcx + OFFSET_SHIFT_THRESHOLD], xmm0
@shift_done:

    movsd xmm0, real8 ptr [rsp + 72] ; halflife_mult
    comisd xmm0, c_zero
    ja @hl_passed
    mov rax, 4014000000000000h ; 5.0 double
    mov qword ptr [rcx + OFFSET_HALFLIFE_MULT], rax
    jmp @hl_done
@hl_passed:
    movsd real8 ptr [rcx + OFFSET_HALFLIFE_MULT], xmm0
@hl_done:

    movsd xmm0, real8 ptr [rsp + 80] ; freq_cap
    comisd xmm0, c_zero
    ja @fc_passed
    mov rax, 402E000000000000h ; 15.0 double
    mov qword ptr [rcx + OFFSET_FREQ_CAP], rax
    jmp @fc_done
@fc_passed:
    movsd real8 ptr [rcx + OFFSET_FREQ_CAP], xmm0
@fc_done:

    mov rax, qword ptr [rsp + 88] ; seed
    test rax, rax
    jnz @seed_passed
    mov rax, 42
@seed_passed:
    mov qword ptr [rcx + OFFSET_RNG_STATE], rax

    ; Initial baseline hit rate = 0.5
    movsd xmm0, c_half
    movsd real8 ptr [rcx + OFFSET_BASELINE_HIT_RATE], xmm0

    ; Initial weights: w_lru = 0.5, w_lfu = 0.5
    movsd real8 ptr [rcx + OFFSET_W_LRU], xmm0
    movsd real8 ptr [rcx + OFFSET_W_LFU], xmm0

    ; window_size = max(20, 2 * capacity), capped at 256
    mov eax, edx
    shl eax, 1
    cmp eax, 20
    jge @win_ok
    mov eax, 20
@win_ok:
    cmp eax, 256
    jle @win_bounded
    mov eax, 256
@win_bounded:
    mov dword ptr [rcx + OFFSET_WINDOW_SIZE], eax

    ; lru_head = -1, lru_tail = -1
    mov dword ptr [rcx + OFFSET_LRU_HEAD], -1
    mov dword ptr [rcx + OFFSET_LRU_TAIL], -1

@init_ret:
    pop rbx
    pop rsi
    pop rdi
    ret
smartcache_init ENDP


; ==============================================================================
; smartcache_reset
; C Prototype: void smartcache_reset(SmartCacheState* state);
; ==============================================================================
smartcache_reset PROC
    push rdi
    push rsi
    push rbx

    test rcx, rcx
    jz @reset_ret

    ; Reset scalar counters
    mov qword ptr [rcx + OFFSET_CURRENT_STEP], 0
    mov qword ptr [rcx + OFFSET_HITS], 0
    mov qword ptr [rcx + OFFSET_MISSES], 0
    mov qword ptr [rcx + OFFSET_EVICTIONS], 0
    mov qword ptr [rcx + OFFSET_GHOST_HITS_LRU], 0
    mov qword ptr [rcx + OFFSET_GHOST_HITS_LFU], 0
    mov qword ptr [rcx + OFFSET_EXPIRED_GHOST], 0
    mov dword ptr [rcx + OFFSET_CURRENT_SIZE], 0
    mov dword ptr [rcx + OFFSET_LRU_HEAD], -1
    mov dword ptr [rcx + OFFSET_LRU_TAIL], -1
    mov dword ptr [rcx + OFFSET_LAST_EXPERT], 0
    mov dword ptr [rcx + OFFSET_IS_SHIFT_ACTIVE], 0

    ; Reset weights to 0.5
    movsd xmm0, c_half
    movsd real8 ptr [rcx + OFFSET_W_LRU], xmm0
    movsd real8 ptr [rcx + OFFSET_W_LFU], xmm0
    movsd real8 ptr [rcx + OFFSET_BASELINE_HIT_RATE], xmm0

    ; Reset outcome window
    mov dword ptr [rcx + OFFSET_WINDOW_HEAD], 0
    mov dword ptr [rcx + OFFSET_WINDOW_COUNT], 0
    mov dword ptr [rcx + OFFSET_WINDOW_HIT_COUNT], 0
    mov dword ptr [rcx + OFFSET_RECENT_GHOST_HEAD], 0
    mov dword ptr [rcx + OFFSET_RECENT_GHOST_COUNT], 0

    ; Clear hash table
    lea rdi, [rcx + OFFSET_HASH_TABLE]
    xor eax, eax
    mov ecx, 2048
    rep stosd
    sub rdi, 2048 * 4
    sub rdi, OFFSET_HASH_TABLE
    mov rcx, rdi

    ; Clear slots
    lea rdi, [rcx + OFFSET_SLOTS]
    xor eax, eax
    mov ecx, (1024 * SLOT_SIZE) / 8
    rep stosq
    sub rdi, 1024 * SLOT_SIZE
    sub rdi, OFFSET_SLOTS
    mov rcx, rdi

    ; Clear ghosts
    mov dword ptr [rcx + OFFSET_GHOST_LRU_COUNT], 0
    mov dword ptr [rcx + OFFSET_GHOST_LRU_HEAD], 0
    lea rdi, [rcx + OFFSET_GHOST_LRU]
    xor eax, eax
    mov ecx, (1024 * GHOST_SLOT_SIZE) / 8
    rep stosq
    sub rdi, 1024 * GHOST_SLOT_SIZE
    sub rdi, OFFSET_GHOST_LRU
    mov rcx, rdi

    mov dword ptr [rcx + OFFSET_GHOST_LFU_COUNT], 0
    mov dword ptr [rcx + OFFSET_GHOST_LFU_HEAD], 0
    lea rdi, [rcx + OFFSET_GHOST_LFU]
    xor eax, eax
    mov ecx, (1024 * GHOST_SLOT_SIZE) / 8
    rep stosq

@reset_ret:
    pop rbx
    pop rsi
    pop rdi
    ret
smartcache_reset ENDP


; ==============================================================================
; Helper: asm_purge_ghost_key
; ==============================================================================
asm_purge_ghost_key PROC
    test r9d, r9d
    jle @p_ret
    lea rax, [rcx + r8]
    xor r10d, r10d
@p_loop:
    cmp dword ptr [rax + GHOST_IS_VALID], 1
    jne @p_next
    cmp qword ptr [rax + GHOST_KEY], rdx
    jne @p_next
    mov dword ptr [rax + GHOST_IS_VALID], 0
@p_next:
    add rax, GHOST_SLOT_SIZE
    inc r10d
    cmp r10d, r9d
    jl @p_loop
@p_ret:
    ret
asm_purge_ghost_key ENDP


; ==============================================================================
; Helper: asm_normalize_weights
; ==============================================================================
asm_normalize_weights PROC
    movsd xmm0, real8 ptr [rcx + OFFSET_W_LRU]
    movsd xmm1, real8 ptr [rcx + OFFSET_W_LFU]
    movsd xmm2, real8 ptr [rcx + OFFSET_EPSILON_MIN]

    maxsd xmm0, xmm2
    maxsd xmm1, xmm2

    movsd xmm3, xmm0
    addsd xmm3, xmm1

    comisd xmm3, c_zero
    jbe @norm_fallback
    ucomisd xmm3, xmm3
    jp @norm_fallback

    divsd xmm0, xmm3
    divsd xmm1, xmm3
    movsd real8 ptr [rcx + OFFSET_W_LRU], xmm0
    movsd real8 ptr [rcx + OFFSET_W_LFU], xmm1
    ret

@norm_fallback:
    movsd xmm0, c_half
    movsd real8 ptr [rcx + OFFSET_W_LRU], xmm0
    movsd real8 ptr [rcx + OFFSET_W_LFU], xmm0
    ret
asm_normalize_weights ENDP


; ==============================================================================
; Helper: asm_compute_utility
; ==============================================================================
asm_compute_utility PROC
    push rbx
    sub rsp, 32

    mov eax, edx
    imul rax, SLOT_SIZE
    lea rbx, [rcx + OFFSET_SLOTS + rax]

    cvtsi2sd xmm0, qword ptr [rbx + SLOT_FREQ]
    movsd xmm1, real8 ptr [rcx + OFFSET_FREQ_CAP]
    minsd xmm0, xmm1

    mov rax, qword ptr [rcx + OFFSET_CURRENT_STEP]
    sub rax, qword ptr [rbx + SLOT_LAST_ACCESS]
    cmp rax, 0
    jge @age_pos
    xor eax, eax
@age_pos:
    cvtsi2sd xmm1, rax

    cvtsi2sd xmm2, dword ptr [rcx + OFFSET_CAPACITY]
    mulsd xmm2, real8 ptr [rcx + OFFSET_HALFLIFE_MULT]

    cmp dword ptr [rcx + OFFSET_IS_SHIFT_ACTIVE], 0
    je @hl_bound
    cvtsi2sd xmm3, dword ptr [rcx + OFFSET_CAPACITY]
    mov rax, 3FF8000000000000h ; 1.5
    movq xmm4, rax
    mulsd xmm3, xmm4
    maxsd xmm3, c_two
    minsd xmm2, xmm3

@hl_bound:
    maxsd xmm2, c_two

    movsd xmm3, c_ln2
    mulsd xmm3, xmm1
    divsd xmm3, xmm2
    xorpd xmm4, xmm4
    subsd xmm4, xmm3

    movsd real8 ptr [rsp], xmm0

    movsd xmm0, xmm4
    call asm_fast_exp

    mulsd xmm0, real8 ptr [rsp]

    add rsp, 32
    pop rbx
    ret
asm_compute_utility ENDP


; ==============================================================================
; smartcache_choose_victim
; ==============================================================================
smartcache_choose_victim PROC
    push rbx
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64

    mov r12, rcx
    mov r13, rdx
    mov r14, r8

    ; LRU victim: slot at lru_head
    mov r15d, dword ptr [r12 + OFFSET_LRU_HEAD]
    cmp r15d, 0
    jge @lru_v_ok
    xor r15d, r15d
@lru_v_ok:

    ; LFU victim: slot with min freq (tie-break with last_access_step)
    mov ebx, -1
    mov rdi, 7FFFFFFFFFFFFFFFh
    mov rsi, 7FFFFFFFFFFFFFFFh

    mov r8d, dword ptr [r12 + OFFSET_CAPACITY]
    xor r9d, r9d
@lfu_scan:
    mov eax, r9d
    imul rax, SLOT_SIZE
    lea r10, [r12 + OFFSET_SLOTS + rax]

    cmp dword ptr [r10 + SLOT_IS_VALID], 1
    jne @lfu_next

    mov r11, qword ptr [r10 + SLOT_FREQ]
    cmp r11, rdi
    jl @lfu_new_min
    jg @lfu_next
    mov rax, qword ptr [r10 + SLOT_LAST_ACCESS]
    cmp rax, rsi
    jl @lfu_new_min
    jmp @lfu_next

@lfu_new_min:
    mov rdi, r11
    mov rsi, qword ptr [r10 + SLOT_LAST_ACCESS]
    mov ebx, r9d

@lfu_next:
    inc r9d
    cmp r9d, r8d
    jl @lfu_scan

    cmp ebx, 0
    jge @lfu_v_ok
    mov ebx, r15d
@lfu_v_ok:

    ; Consensus Case
    cmp r15d, ebx
    jne @disagreement

    mov dword ptr [rsp + 32], 3
    mov dword ptr [rsp + 36], r15d
    jmp @victim_selected

@disagreement:
    mov rcx, r12
    mov edx, r15d
    call asm_compute_utility
    movsd real8 ptr [rsp + 40], xmm0

    mov rcx, r12
    mov edx, ebx
    call asm_compute_utility
    movsd real8 ptr [rsp + 48], xmm0

    cmp dword ptr [r12 + OFFSET_DETERMINISTIC], 1
    jne @stochastic_choice

    movsd xmm0, real8 ptr [rsp + 48]
    comisd xmm0, real8 ptr [rsp + 40]
    jbe @choose_lfu
    jmp @choose_lru

@stochastic_choice:
    movsd xmm0, real8 ptr [rsp + 40]
    subsd xmm0, real8 ptr [rsp + 48]
    mulsd xmm0, c_logistic_k
    maxsd xmm0, c_clamp_min
    minsd xmm0, c_clamp_max
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    movsd xmm0, xmm1
    call asm_fast_exp
    addsd xmm0, c_one
    movsd xmm1, c_one
    divsd xmm1, xmm0

    mov rcx, r12
    movsd real8 ptr [rsp + 56], xmm1
    call asm_xorshift64
    movsd xmm1, real8 ptr [rsp + 56]
    comisd xmm0, xmm1
    jb @choose_lfu
    jmp @choose_lru

@choose_lfu:
    mov dword ptr [rsp + 32], 2
    mov dword ptr [rsp + 36], ebx
    jmp @victim_selected

@choose_lru:
    mov dword ptr [rsp + 32], 1
    mov dword ptr [rsp + 36], r15d

@victim_selected:
    mov eax, dword ptr [rsp + 36]
    test r13, r13
    jz @no_slot_out
    mov dword ptr [r13], eax
@no_slot_out:

    mov edx, dword ptr [rsp + 32]
    test r14, r14
    jz @no_exp_out
    mov dword ptr [r14], edx
@no_exp_out:

    imul rax, SLOT_SIZE
    lea r10, [r12 + OFFSET_SLOTS + rax]
    mov rax, qword ptr [r10 + SLOT_KEY]

    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rsi
    pop rdi
    pop rbx
    ret
smartcache_choose_victim ENDP


; ==============================================================================
; smartcache_access
; ==============================================================================
smartcache_access PROC
    push rbx
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 80

    mov r12, rcx
    mov r13, rdx
    mov r14, r8
    mov r15, r9

    mov rax, qword ptr [rsp + 176]
    mov qword ptr [rsp + 32], rax

    ; 1. Step Increment
    inc qword ptr [r12 + OFFSET_CURRENT_STEP]
    mov rsi, qword ptr [r12 + OFFSET_CURRENT_STEP]

    ; 2. Cache Lookup
    mov ebx, dword ptr [r12 + OFFSET_CAPACITY]
    xor edi, edi
    mov eax, -1

@lookup_loop:
    cmp edi, ebx
    jge @lookup_done

    mov r8d, edi
    imul r8, SLOT_SIZE
    lea r10, [r12 + OFFSET_SLOTS + r8]

    cmp dword ptr [r10 + SLOT_IS_VALID], 1
    jne @lookup_next
    cmp qword ptr [r10 + SLOT_KEY], r13
    je @lookup_hit

@lookup_next:
    inc edi
    jmp @lookup_loop

@lookup_hit:
    mov eax, edi

@lookup_done:
    cmp eax, -1
    je @handle_miss

    ; ==========================================================================
    ; CACHE HIT HANDLER
    ; ==========================================================================
    inc qword ptr [r12 + OFFSET_HITS]
    mov edi, eax

    imul rax, SLOT_SIZE
    lea r10, [r12 + OFFSET_SLOTS + rax]

    mov qword ptr [r10 + SLOT_LAST_ACCESS], rsi
    inc qword ptr [r10 + SLOT_FREQ]

    ; Update LRU list
    mov eax, dword ptr [r12 + OFFSET_LRU_TAIL]
    cmp edi, eax
    je @hit_lru_done

    ; Unlink edi
    mov r8d, dword ptr [r10 + SLOT_PREV_LRU]
    mov r9d, dword ptr [r10 + SLOT_NEXT_LRU]

    cmp r8d, -1
    je @hit_head_unlink
    imul rax, r8, SLOT_SIZE
    mov dword ptr [r12 + OFFSET_SLOTS + rax + SLOT_NEXT_LRU], r9d
    jmp @hit_check_next
@hit_head_unlink:
    mov dword ptr [r12 + OFFSET_LRU_HEAD], r9d

@hit_check_next:
    cmp r9d, -1
    je @hit_tail_unlink
    imul rax, r9, SLOT_SIZE
    mov dword ptr [r12 + OFFSET_SLOTS + rax + SLOT_PREV_LRU], r8d
    jmp @hit_relink_tail
@hit_tail_unlink:
    mov dword ptr [r12 + OFFSET_LRU_TAIL], r8d

@hit_relink_tail:
    mov eax, dword ptr [r12 + OFFSET_LRU_TAIL]
    mov dword ptr [r10 + SLOT_PREV_LRU], eax
    mov dword ptr [r10 + SLOT_NEXT_LRU], -1
    cmp eax, -1
    je @hit_empty_tail
    imul rax, rax, SLOT_SIZE
    mov dword ptr [r12 + OFFSET_SLOTS + rax + SLOT_NEXT_LRU], edi
    jmp @hit_set_tail
@hit_empty_tail:
    mov dword ptr [r12 + OFFSET_LRU_HEAD], edi
@hit_set_tail:
    mov dword ptr [r12 + OFFSET_LRU_TAIL], edi

@hit_lru_done:
    ; Ghost cache purge
    mov rcx, r12
    mov rdx, r13
    mov r8, OFFSET_GHOST_LRU
    mov r9d, dword ptr [r12 + OFFSET_GHOST_CAPACITY]
    call asm_purge_ghost_key

    mov rcx, r12
    mov rdx, r13
    mov r8, OFFSET_GHOST_LFU
    mov r9d, dword ptr [r12 + OFFSET_GHOST_CAPACITY]
    call asm_purge_ghost_key

    ; Update circular window with O(1) hit count
    mov eax, dword ptr [r12 + OFFSET_WINDOW_HEAD]
    mov edx, dword ptr [r12 + OFFSET_WINDOW_COUNT]
    cmp edx, dword ptr [r12 + OFFSET_WINDOW_SIZE]
    jl @win_hit_not_full
    movzx edx, byte ptr [r12 + OFFSET_WINDOW_HITS + rax]
    sub dword ptr [r12 + OFFSET_WINDOW_HIT_COUNT], edx
@win_hit_not_full:
    mov byte ptr [r12 + OFFSET_WINDOW_HITS + rax], 1
    inc dword ptr [r12 + OFFSET_WINDOW_HIT_COUNT]
    mov qword ptr [r12 + OFFSET_WINDOW_KEYS + rax * 8], r13

    inc eax
    cmp eax, dword ptr [r12 + OFFSET_WINDOW_SIZE]
    jl @win_hit_wrap_ok
    xor eax, eax
@win_hit_wrap_ok:
    mov dword ptr [r12 + OFFSET_WINDOW_HEAD], eax
    mov eax, dword ptr [r12 + OFFSET_WINDOW_COUNT]
    cmp eax, dword ptr [r12 + OFFSET_WINDOW_SIZE]
    jge @win_hit_cnt_ok
    inc dword ptr [r12 + OFFSET_WINDOW_COUNT]
@win_hit_cnt_ok:

    ; Baseline hit rate EMA: 0.95 * rate + 0.05 * 1.0
    movsd xmm0, real8 ptr [r12 + OFFSET_BASELINE_HIT_RATE]
    mulsd xmm0, c_ema_beta
    addsd xmm0, c_ema_alpha
    movsd real8 ptr [r12 + OFFSET_BASELINE_HIT_RATE], xmm0

    ; Outputs
    test r14, r14
    jz @hit_no_is_hit
    mov dword ptr [r14], 1
@hit_no_is_hit:

    test r15, r15
    jz @hit_no_evicted
    mov qword ptr [r15], 0
@hit_no_evicted:

    mov rax, qword ptr [rsp + 32]
    test rax, rax
    jz @hit_no_exp
    mov dword ptr [rax], 0
@hit_no_exp:

    mov eax, 1
    jmp @access_ret

    ; ==========================================================================
    ; CACHE MISS HANDLER
    ; ==========================================================================
@handle_miss:
    inc qword ptr [r12 + OFFSET_MISSES]

    ; Update circular window with O(1) hit count
    mov eax, dword ptr [r12 + OFFSET_WINDOW_HEAD]
    mov edx, dword ptr [r12 + OFFSET_WINDOW_COUNT]
    cmp edx, dword ptr [r12 + OFFSET_WINDOW_SIZE]
    jl @win_miss_not_full
    movzx edx, byte ptr [r12 + OFFSET_WINDOW_HITS + rax]
    sub dword ptr [r12 + OFFSET_WINDOW_HIT_COUNT], edx
@win_miss_not_full:
    mov byte ptr [r12 + OFFSET_WINDOW_HITS + rax], 0
    mov qword ptr [r12 + OFFSET_WINDOW_KEYS + rax * 8], r13

    inc eax
    cmp eax, dword ptr [r12 + OFFSET_WINDOW_SIZE]
    jl @win_miss_wrap_ok
    xor eax, eax
@win_miss_wrap_ok:
    mov dword ptr [r12 + OFFSET_WINDOW_HEAD], eax
    mov eax, dword ptr [r12 + OFFSET_WINDOW_COUNT]
    cmp eax, dword ptr [r12 + OFFSET_WINDOW_SIZE]
    jge @win_miss_cnt_ok
    inc dword ptr [r12 + OFFSET_WINDOW_COUNT]
@win_miss_cnt_ok:

    ; Workload shift evaluation (O(1))
    mov eax, dword ptr [r12 + OFFSET_WINDOW_COUNT]
    cmp eax, dword ptr [r12 + OFFSET_CAPACITY]
    jl @shift_eval_done

    cvtsi2sd xmm0, dword ptr [r12 + OFFSET_WINDOW_HIT_COUNT]
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1 ; recent_hr in O(1)!

    ; Condition 1: Hit rate drop below baseline
    movsd xmm1, real8 ptr [r12 + OFFSET_BASELINE_HIT_RATE]
    subsd xmm1, xmm0
    comisd xmm1, real8 ptr [r12 + OFFSET_SHIFT_THRESHOLD]
    jae @trigger_shift

    ; Condition 2: Sequential scan / burst detection (recent_hr < 0.15)
    mov rax, 3FC3333333333333h ; 0.15 double
    movq xmm2, rax
    comisd xmm0, xmm2
    jb @trigger_shift

    ; Condition 3: Sustained regret on dominant expert
    mov r8d, dword ptr [r12 + OFFSET_RECENT_GHOST_COUNT]
    cmp r8d, 2
    jge @trigger_shift

    mov dword ptr [r12 + OFFSET_IS_SHIFT_ACTIVE], 0
    jmp @shift_eval_done

@trigger_shift:
    mov dword ptr [r12 + OFFSET_IS_SHIFT_ACTIVE], 1

@shift_eval_done:
    ; Baseline hit rate EMA decay: 0.95 * rate
    movsd xmm0, real8 ptr [r12 + OFFSET_BASELINE_HIT_RATE]
    mulsd xmm0, c_ema_beta
    movsd real8 ptr [r12 + OFFSET_BASELINE_HIT_RATE], xmm0

    ; --------------------------------------------------------------------------
    ; Regret Attribution: Check Dual Ghost Caches
    ; --------------------------------------------------------------------------
    mov ebx, dword ptr [r12 + OFFSET_GHOST_CAPACITY]
    xor edi, edi
    mov dword ptr [rsp + 40], 0

    movsd xmm0, real8 ptr [r12 + OFFSET_W_LRU]
    comisd xmm0, real8 ptr [r12 + OFFSET_W_LFU]
    jae @dom_lru
    mov dword ptr [rsp + 44], 2
    jmp @ghost_lru_search
@dom_lru:
    mov dword ptr [rsp + 44], 1

@ghost_lru_search:
    cmp edi, ebx
    jge @ghost_lru_done
    mov r8d, edi
    imul r8, GHOST_SLOT_SIZE
    lea r10, [r12 + OFFSET_GHOST_LRU + r8]

    cmp dword ptr [r10 + GHOST_IS_VALID], 1
    jne @ghost_lru_next
    cmp qword ptr [r10 + GHOST_KEY], r13
    jne @ghost_lru_next

    mov dword ptr [r10 + GHOST_IS_VALID], 0
    mov rax, rsi
    sub rax, qword ptr [r10 + GHOST_EVICT_STEP]
    cmp eax, dword ptr [r12 + OFFSET_GHOST_TTL]
    jg @ghost_lru_expired

    inc qword ptr [r12 + OFFSET_GHOST_HITS_LRU]
    movsd xmm0, real8 ptr [r12 + OFFSET_LEARNING_RATE]
    cmp dword ptr [r12 + OFFSET_IS_SHIFT_ACTIVE], 1
    je @lru_lr_full
    cmp dword ptr [rsp + 44], 1
    jne @lru_lr_full
    mulsd xmm0, real8 ptr [r12 + OFFSET_DAMPING_FACTOR]
@lru_lr_full:
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    movsd xmm0, xmm1
    call asm_fast_exp
    mulsd xmm0, real8 ptr [r12 + OFFSET_W_LRU]
    movsd real8 ptr [r12 + OFFSET_W_LRU], xmm0
    mov dword ptr [rsp + 40], 1
    jmp @ghost_lru_done

@ghost_lru_expired:
    inc qword ptr [r12 + OFFSET_EXPIRED_GHOST]

@ghost_lru_next:
    inc edi
    jmp @ghost_lru_search

@ghost_lru_done:

    xor edi, edi
@ghost_lfu_search:
    cmp edi, ebx
    jge @ghost_lfu_done
    mov r8d, edi
    imul r8, GHOST_SLOT_SIZE
    lea r10, [r12 + OFFSET_GHOST_LFU + r8]

    cmp dword ptr [r10 + GHOST_IS_VALID], 1
    jne @ghost_lfu_next
    cmp qword ptr [r10 + GHOST_KEY], r13
    jne @ghost_lfu_next

    mov dword ptr [r10 + GHOST_IS_VALID], 0
    mov rax, rsi
    sub rax, qword ptr [r10 + GHOST_EVICT_STEP]
    cmp eax, dword ptr [r12 + OFFSET_GHOST_TTL]
    jg @ghost_lfu_expired

    inc qword ptr [r12 + OFFSET_GHOST_HITS_LFU]
    movsd xmm0, real8 ptr [r12 + OFFSET_LEARNING_RATE]
    cmp dword ptr [r12 + OFFSET_IS_SHIFT_ACTIVE], 1
    je @lfu_lr_full
    cmp dword ptr [rsp + 44], 2
    jne @lfu_lr_full
    mulsd xmm0, real8 ptr [r12 + OFFSET_DAMPING_FACTOR]
@lfu_lr_full:
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    movsd xmm0, xmm1
    call asm_fast_exp
    mulsd xmm0, real8 ptr [r12 + OFFSET_W_LFU]
    movsd real8 ptr [r12 + OFFSET_W_LFU], xmm0
    mov dword ptr [rsp + 40], 1
    jmp @ghost_lfu_done

@ghost_lfu_expired:
    inc qword ptr [r12 + OFFSET_EXPIRED_GHOST]

@ghost_lfu_next:
    inc edi
    jmp @ghost_lfu_search

@ghost_lfu_done:

    mov rcx, r12
    mov rdx, r13
    mov r8, OFFSET_GHOST_LRU
    mov r9d, dword ptr [r12 + OFFSET_GHOST_CAPACITY]
    call asm_purge_ghost_key

    mov rcx, r12
    mov rdx, r13
    mov r8, OFFSET_GHOST_LFU
    mov r9d, dword ptr [r12 + OFFSET_GHOST_CAPACITY]
    call asm_purge_ghost_key

    cmp dword ptr [rsp + 40], 1
    jne @weights_normalized
    mov rcx, r12
    call asm_normalize_weights
@weights_normalized:

    ; --------------------------------------------------------------------------
    ; Eviction & Insertion
    ; --------------------------------------------------------------------------
    mov eax, dword ptr [r12 + OFFSET_CURRENT_SIZE]
    cmp eax, dword ptr [r12 + OFFSET_CAPACITY]
    jl @insert_free_slot

    inc qword ptr [r12 + OFFSET_EVICTIONS]

    mov rcx, r12
    lea rdx, [rsp + 48]
    lea r8, [rsp + 52]
    call smartcache_choose_victim
    mov qword ptr [rsp + 56], rax

    mov edi, dword ptr [rsp + 48]
    mov ebx, dword ptr [rsp + 52]
    mov dword ptr [r12 + OFFSET_LAST_EXPERT], ebx

    mov rdx, qword ptr [rsp + 56]

    cmp ebx, 1
    je @ghost_add_lru
    cmp ebx, 2
    je @ghost_add_lfu
    jmp @eviction_unlinked

@ghost_add_lru:
    mov rcx, r12
    mov r8, OFFSET_GHOST_LFU
    mov r9d, dword ptr [r12 + OFFSET_GHOST_CAPACITY]
    call asm_purge_ghost_key

    mov eax, dword ptr [r12 + OFFSET_GHOST_LRU_HEAD]
    imul rax, GHOST_SLOT_SIZE
    lea r10, [r12 + OFFSET_GHOST_LRU + rax]
    mov qword ptr [r10 + GHOST_KEY], rdx
    mov qword ptr [r10 + GHOST_EVICT_STEP], rsi
    mov dword ptr [r10 + GHOST_IS_VALID], 1
    inc dword ptr [r12 + OFFSET_GHOST_LRU_HEAD]
    mov eax, dword ptr [r12 + OFFSET_GHOST_LRU_HEAD]
    cmp eax, dword ptr [r12 + OFFSET_GHOST_CAPACITY]
    jl @g_lru_head_ok
    mov dword ptr [r12 + OFFSET_GHOST_LRU_HEAD], 0
@g_lru_head_ok:
    jmp @eviction_unlinked

@ghost_add_lfu:
    mov rcx, r12
    mov r8, OFFSET_GHOST_LRU
    mov r9d, dword ptr [r12 + OFFSET_GHOST_CAPACITY]
    call asm_purge_ghost_key

    mov eax, dword ptr [r12 + OFFSET_GHOST_LFU_HEAD]
    imul rax, GHOST_SLOT_SIZE
    lea r10, [r12 + OFFSET_GHOST_LFU + rax]
    mov qword ptr [r10 + GHOST_KEY], rdx
    mov qword ptr [r10 + GHOST_EVICT_STEP], rsi
    mov dword ptr [r10 + GHOST_IS_VALID], 1
    inc dword ptr [r12 + OFFSET_GHOST_LFU_HEAD]
    mov eax, dword ptr [r12 + OFFSET_GHOST_LFU_HEAD]
    cmp eax, dword ptr [r12 + OFFSET_GHOST_CAPACITY]
    jl @g_lfu_head_ok
    mov dword ptr [r12 + OFFSET_GHOST_LFU_HEAD], 0
@g_lfu_head_ok:

@eviction_unlinked:
    imul rax, rdi, SLOT_SIZE
    lea r10, [r12 + OFFSET_SLOTS + rax]
    mov r8d, dword ptr [r10 + SLOT_PREV_LRU]
    mov r9d, dword ptr [r10 + SLOT_NEXT_LRU]

    cmp r8d, -1
    je @evict_head_unlink
    imul rax, r8, SLOT_SIZE
    mov dword ptr [r12 + OFFSET_SLOTS + rax + SLOT_NEXT_LRU], r9d
    jmp @evict_check_next
@evict_head_unlink:
    mov dword ptr [r12 + OFFSET_LRU_HEAD], r9d

@evict_check_next:
    cmp r9d, -1
    je @evict_tail_unlink
    imul rax, r9, SLOT_SIZE
    mov dword ptr [r12 + OFFSET_SLOTS + rax + SLOT_PREV_LRU], r8d
    jmp @evict_relinked
@evict_tail_unlink:
    mov dword ptr [r12 + OFFSET_LRU_TAIL], r8d

@evict_relinked:
    jmp @install_new_item

@insert_free_slot:
    mov edi, dword ptr [r12 + OFFSET_CURRENT_SIZE]
    inc dword ptr [r12 + OFFSET_CURRENT_SIZE]
    mov qword ptr [rsp + 56], 0
    mov dword ptr [rsp + 52], 0

@install_new_item:
    imul rax, rdi, SLOT_SIZE
    lea r10, [r12 + OFFSET_SLOTS + rax]
    mov qword ptr [r10 + SLOT_KEY], r13
    mov qword ptr [r10 + SLOT_LAST_ACCESS], rsi
    mov qword ptr [r10 + SLOT_FREQ], 1
    mov dword ptr [r10 + SLOT_IS_VALID], 1

    mov eax, dword ptr [r12 + OFFSET_LRU_TAIL]
    mov dword ptr [r10 + SLOT_PREV_LRU], eax
    mov dword ptr [r10 + SLOT_NEXT_LRU], -1
    cmp eax, -1
    je @ins_empty_tail
    imul rax, rax, SLOT_SIZE
    mov dword ptr [r12 + OFFSET_SLOTS + rax + SLOT_NEXT_LRU], edi
    jmp @ins_set_tail
@ins_empty_tail:
    mov dword ptr [r12 + OFFSET_LRU_HEAD], edi
@ins_set_tail:
    mov dword ptr [r12 + OFFSET_LRU_TAIL], edi

    test r14, r14
    jz @miss_no_is_hit
    mov dword ptr [r14], 0
@miss_no_is_hit:

    test r15, r15
    jz @miss_no_evicted
    mov rax, qword ptr [rsp + 56]
    mov qword ptr [r15], rax
@miss_no_evicted:

    mov rax, qword ptr [rsp + 32]
    test rax, rax
    jz @miss_no_exp
    mov edx, dword ptr [rsp + 52]
    mov dword ptr [rax], edx
@miss_no_exp:

    xor eax, eax

@access_ret:
    add rsp, 80
    pop r15
    pop r14
    pop r13
    pop r12
    pop rsi
    pop rdi
    pop rbx
    ret
smartcache_access ENDP


; ==============================================================================
; smartcache_get_weights
; ==============================================================================
smartcache_get_weights PROC
    test rdx, rdx
    jz @gw_no_lru
    movsd xmm0, real8 ptr [rcx + OFFSET_W_LRU]
    movsd real8 ptr [rdx], xmm0
@gw_no_lru:
    test r8, r8
    jz @gw_no_lfu
    movsd xmm1, real8 ptr [rcx + OFFSET_W_LFU]
    movsd real8 ptr [r8], xmm1
@gw_no_lfu:
    ret
smartcache_get_weights ENDP


; ==============================================================================
; smartcache_run_trace
; ==============================================================================
smartcache_run_trace PROC
    push rbx
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 104

    mov r12, rcx
    mov r13, rdx
    mov r14, r8
    mov r15, r9

    mov rax, qword ptr [rsp + 200]
    mov qword ptr [rsp + 32], rax
    mov rax, qword ptr [rsp + 208]
    mov qword ptr [rsp + 40], rax
    mov rax, qword ptr [rsp + 216]
    mov qword ptr [rsp + 48], rax

    ; Start RDTSC timer
    lfence
    rdtsc
    shl rdx, 32
    or rax, rdx
    mov qword ptr [rsp + 56], rax

    xor rsi, rsi
@trace_loop:
    cmp rsi, r14
    jge @trace_done

    mov rdx, qword ptr [r13 + rsi * 8]
    mov rcx, r12
    xor r8, r8
    xor r9, r9
    mov qword ptr [rsp + 64], 0
    lea rax, [rsp + 64]
    mov qword ptr [rsp + 32 + 8], rax
    call smartcache_access

    inc rsi
    jmp @trace_loop

@trace_done:
    ; Stop RDTSC timer
    lfence
    rdtsc
    shl rdx, 32
    or rax, rdx
    sub rax, qword ptr [rsp + 56]
    mov qword ptr [rsp + 72], rax

    test r15, r15
    jz @no_hits_out
    mov rax, qword ptr [r12 + OFFSET_HITS]
    mov qword ptr [r15], rax
@no_hits_out:

    mov rbx, qword ptr [rsp + 32]
    test rbx, rbx
    jz @no_misses_out
    mov rax, qword ptr [r12 + OFFSET_MISSES]
    mov qword ptr [rbx], rax
@no_misses_out:

    mov rbx, qword ptr [rsp + 40]
    test rbx, rbx
    jz @no_evict_out
    mov rax, qword ptr [r12 + OFFSET_EVICTIONS]
    mov qword ptr [rbx], rax
@no_evict_out:

    mov rbx, qword ptr [rsp + 48]
    test rbx, rbx
    jz @no_ns_out
    test r14, r14
    jle @no_ns_out

    cvtsi2sd xmm0, qword ptr [rsp + 72]
    cvtsi2sd xmm1, r14
    divsd xmm0, xmm1
    mov rax, 3FD5555555555555h ; 1/3.0 GHz approx
    movq xmm1, rax
    mulsd xmm0, xmm1
    movsd real8 ptr [rbx], xmm0
@no_ns_out:

    add rsp, 104
    pop r15
    pop r14
    pop r13
    pop r12
    pop rsi
    pop rdi
    pop rbx
    ret
smartcache_run_trace ENDP

END
