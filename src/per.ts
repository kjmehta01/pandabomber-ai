// Prioritized Experience Replay (Schaul et al. 2016), proportional, sum-tree for
// O(log N) sample + update.
//
// Tree is padded to the next power of 2 so leaves form a perfect binary tree;
// extra slots stay at priority 0 and are unreachable from sampling. Tree node i
// has children 2i and 2i+1; leaves at [leafBase, leafBase + treeCap); internal
// nodes hold their subtree sum; total priority is tree[1].
//
// Stratified sampling + IS weights normalized by batch max (standard PER trick:
// the largest weight is 1.0 so loss scale stays comparable to uniform sampling).

import { OBS_SIZE } from './observation';

export interface PERSample {
    s: Float32Array;
    a: Int32Array;
    r: Float32Array;
    sNext: Float32Array;
    done: Uint8Array;
    nStep: Int32Array;
    indices: Int32Array;     // pass back to updatePriorities()
    isWeights: Float32Array;
}

export class PrioritizedReplayBuffer {
    readonly cap: number;
    size = 0;
    private head = 0;

    private obs: Float32Array[];
    private next: Float32Array[];
    private act: Int32Array;
    private rew: Float32Array;
    private doneArr: Uint8Array;
    private nStepArr: Int32Array;

    private treeCap: number;     // power of 2 ≥ cap
    private leafBase: number;
    private tree: Float64Array;

    private maxPrio = 1.0;       // initial floor; rises with observed |TD|^α
    private readonly alpha: number;
    private readonly eps: number;

    constructor(capacity: number, alpha = 0.6, eps = 1e-6) {
        this.cap = capacity;
        this.obs = new Array(capacity);
        this.next = new Array(capacity);
        this.act = new Int32Array(capacity);
        this.rew = new Float32Array(capacity);
        this.doneArr = new Uint8Array(capacity);
        this.nStepArr = new Int32Array(capacity);

        let p = 1;
        while (p < capacity) p <<= 1;
        this.treeCap = p;
        this.leafBase = p;
        this.tree = new Float64Array(2 * p);
        this.alpha = alpha;
        this.eps = eps;
    }

    push(s: Float32Array, a: number, r: number, sNext: Float32Array, done: boolean, nStep: number): void {
        const idx = this.head;
        this.obs[idx] = s;
        this.next[idx] = sNext;
        this.act[idx] = a;
        this.rew[idx] = r;
        this.doneArr[idx] = done ? 1 : 0;
        this.nStepArr[idx] = nStep;

        // Seed at maxPrio so every new transition is sampled at least once before
        // its priority is overwritten by its first observed TD error.
        this._setPriority(idx, this.maxPrio);

        this.head = (this.head + 1) % this.cap;
        if (this.size < this.cap) this.size++;
    }

    sample(n: number, beta: number): PERSample {
        const total = this.tree[1];
        if (total <= 0 || this.size === 0) {
            throw new Error('PER.sample called on empty buffer');
        }
        const seg = total / n;

        const sBuf = new Float32Array(n * OBS_SIZE);
        const sNextBuf = new Float32Array(n * OBS_SIZE);
        const aBuf = new Int32Array(n);
        const rBuf = new Float32Array(n);
        const dBuf = new Uint8Array(n);
        const nStepBuf = new Int32Array(n);
        const indices = new Int32Array(n);
        const isW = new Float32Array(n);
        const probs = new Float32Array(n);

        for (let i = 0; i < n; i++) {
            const lo = seg * i;
            const hi = seg * (i + 1);
            const v = lo + Math.random() * (hi - lo);
            const idx = this._retrieve(v);
            const treeIdx = this.leafBase + idx;
            const p = this.tree[treeIdx];
            indices[i] = idx;
            probs[i] = p / total;
            sBuf.set(this.obs[idx], i * OBS_SIZE);
            sNextBuf.set(this.next[idx], i * OBS_SIZE);
            aBuf[i] = this.act[idx];
            rBuf[i] = this.rew[idx];
            dBuf[i] = this.doneArr[idx];
            nStepBuf[i] = this.nStepArr[idx];
        }

        // w_i = (N · P(i))^(−β), normalized by batch max.
        const N = this.size;
        let maxW = 0;
        for (let i = 0; i < n; i++) {
            const w = Math.pow(N * Math.max(probs[i], 1e-12), -beta);
            isW[i] = w;
            if (w > maxW) maxW = w;
        }
        if (maxW > 0) {
            for (let i = 0; i < n; i++) isW[i] /= maxW;
        }

        return { s: sBuf, a: aBuf, r: rBuf, sNext: sNextBuf, done: dBuf, nStep: nStepBuf, indices, isWeights: isW };
    }

    // absTdErrors[i] is |TD| for sample i; this method applies α and ε.
    updatePriorities(indices: Int32Array, absTdErrors: Float32Array): void {
        for (let k = 0; k < indices.length; k++) {
            const prio = absTdErrors[k] + this.eps;
            const p = Math.pow(prio, this.alpha);
            this._setPriority(indices[k], p);
            if (p > this.maxPrio) this.maxPrio = p;
        }
    }

    private _setPriority(idx: number, p: number): void {
        const treeIdx = this.leafBase + idx;
        const change = p - this.tree[treeIdx];
        if (change === 0) return;
        this.tree[treeIdx] = p;
        let i = treeIdx >> 1;
        while (i >= 1) {
            this.tree[i] += change;
            i >>= 1;
        }
    }

    private _retrieve(v: number): number {
        let i = 1;
        while (i < this.leafBase) {
            const left = 2 * i;
            const leftP = this.tree[left];
            if (v <= leftP) {
                i = left;
            } else {
                v -= leftP;
                i = left + 1;
            }
        }
        return i - this.leafBase;
    }
}
