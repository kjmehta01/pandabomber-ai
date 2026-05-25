// Observation encoder shared by sim-side training and gameserver-side inference.
// Framework-free (plain Float32Array) so bot2.ts can call it without TF.js.
//
// Spatial layout is BOARD_H × BOARD_W × NUM_CHANNELS in NHWC order so the buffer
// reshapes directly into a conv2d tensor. Scalars are appended after.
//
// Egocentric framing: the view is shifted so self is always at SELF_R, SELF_C
// (board center). Cells whose source falls off the real board are stone-padded
// on channel 0. Spatial dims are unchanged from the raw board, so corner-standing
// agents see ~half the real board as inert stone — accepted trade-off in exchange
// for a fixed self position the conv layers don't have to learn to find.

export const BOARD_H = 13;
export const BOARD_W = 19;
// Center of the encoded view; self position is fixed here.
export const SELF_R = (BOARD_H - 1) >> 1; // 6
export const SELF_C = (BOARD_W - 1) >> 1; // 9

// Spatial channels:
//   0  stone (real walls AND off-board padding)
//   1  wood                2  bomb present
//   3  bomb fuse (1=just placed, 0=about to explode)
//   4  time-to-blast (1=exploding now, 0=>=DANGER_LOOKAHEAD_MS away);
//      includes chain-triggered earlier detonation times
//   5  powerup NUM         6  powerup SPE         7  powerup STR
//   8  self position       9  enemy positions
//  10  knocked enemies: knockMsLeft/KNOCK_DURATION_MS (1=just knocked, 0=recovering now)
//  11  active blast cells: msUntilSafe/ACTIVE_BLAST_DANGER_MS — a bomb that already
//      detonated is still lethal for up to ~300ms (per-cell arrival + 50ms second-check)
//  12  bomb power: power/MAX_POWERUP_TIER on each bomb cell (0 elsewhere)
//  13  ticks-to-reach from self via BFS, normalized: max(0, 1 - msToReach/DANGER_LOOKAHEAD_MS).
//      msToReach = bfsDepth / moveSpeed. Walls and (other) bomb cells block.
//  14  reachable-and-survivable mask: 1 if cell is BFS-reachable AND
//      (blastTime==Inf OR msToReach < blastTime). Hands the agent its dodge plan.
export const NUM_CHANNELS = 15;
// Scalars (all normalized): bombPower/13, maxBombs/13, moveSpeed tier/13,
// placedBombs/maxBombs, woodLeft/100, game phase (0=lots of walls, 1=none),
// safe immediate neighbors / 4 (count of 4-connected neighbors that are reachable
// AND survivable — a "freedom" signal for cornering).
export const NUM_SCALARS = 7;

export const OBS_SPATIAL_SIZE = BOARD_H * BOARD_W * NUM_CHANNELS;
export const OBS_SIZE = OBS_SPATIAL_SIZE + NUM_SCALARS;

const BOMB_FUSE_MS = 3000;
// Covers the full fuse so the danger map lights up at placement time.
const DANGER_LOOKAHEAD_MS = 3000;
const KNOCK_DURATION_MS = 6000;
const MAX_POWERUP_TIER = 13;
const START_MOVE_SPEED = 0.045;
const MOVE_SPEED_INCREMENT = 0.003;
// Mirrors sim.ts CELLS_PER_MS(speed) = speed * 1.667 * 60 / 1000. moveSpeed is NOT
// cells/ms — it's the raw per-frame speed multiplier; convert here for time-based features.
const SPEED_TO_CELLS_PER_MS = 1.667 * 60 / 1000;
// Matches sim.ts and production game.ts:12. A power=10 bomb's far end is +250ms
// past center — material to the agent's dodge plan.
const EXPLOSION_TRAVEL_MS = 25;
// Far end of a power-10 bomb's kill window: 250ms travel + 50ms second-check.
const ACTIVE_BLAST_DANGER_MS = 300;

const DIRS: Array<[number, number]> = [[-1, 0], [1, 0], [0, -1], [0, 1]];

// Minimal interface so this module doesn't depend on Sim; bot2.ts builds its own
// SimView from network state.
export interface ObsView {
    boardH: number;
    boardW: number;
    getCell(r: number, c: number): string;
    bombs: Array<{ row: number; col: number; power: number; fuseRemainingMs: number }>;
    self: { y: number; x: number; bombPower: number; maxBombs: number; moveSpeed: number; placedBombs: number };
    enemies: Array<{ y: number; x: number; alive: boolean; knocked: boolean; knockMsLeft: number }>;
    // Cells currently inside a detonated-bomb kill window; msUntilSafe is the time
    // until the cell's second kill-check fires.
    activeBlasts: Array<{ row: number; col: number; msUntilSafe: number }>;
    woodLeft: number;
}

// NHWC flat index.
function idx(r: number, x: number, c: number): number {
    return r * BOARD_W * NUM_CHANNELS + x * NUM_CHANNELS + c;
}

// Min ms until each cell gets blasted by some pending bomb, accounting for chain
// detonation: bomb A's ray reaching bomb B forces B to detonate at A's arrival
// time. Iterate until effective fuses stabilize (bounded by bomb count).
function computeBlastTimes(view: ObsView): Float32Array {
    const out = new Float32Array(view.boardH * view.boardW);
    out.fill(Infinity);
    if (view.bombs.length === 0) return out;

    // Effective fuse per bomb after chain propagation.
    const fuses: number[] = view.bombs.map(b => b.fuseRemainingMs);
    // Quick lookup of bomb index by cell.
    const bombAt = new Map<number, number>();
    for (let i = 0; i < view.bombs.length; i++) {
        bombAt.set(view.bombs[i].row * view.boardW + view.bombs[i].col, i);
    }

    let changed = true;
    let iter = 0;
    const maxIter = view.bombs.length + 1;
    while (changed && iter++ < maxIter) {
        changed = false;
        for (let bi = 0; bi < view.bombs.length; bi++) {
            const b = view.bombs[bi];
            const bombFuse = fuses[bi];
            for (const [dy, dx] of DIRS) {
                for (let i = 1; i <= b.power; i++) {
                    const r = b.row + dy * i;
                    const c = b.col + dx * i;
                    if (r < 0 || r >= view.boardH || c < 0 || c >= view.boardW) break;
                    const cell = view.getCell(r, c);
                    if (cell === 'S') break;
                    const otherIdx = bombAt.get(r * view.boardW + c);
                    if (otherIdx !== undefined) {
                        const trigger = bombFuse + i * EXPLOSION_TRAVEL_MS;
                        if (trigger < fuses[otherIdx]) {
                            fuses[otherIdx] = trigger;
                            changed = true;
                        }
                    }
                    if (cell === 'W') break;
                }
            }
        }
    }

    // Paint blast times using effective fuses.
    for (let bi = 0; bi < view.bombs.length; bi++) {
        const b = view.bombs[bi];
        const fuse = fuses[bi];
        const mark = (r: number, c: number, t: number) => {
            const k = r * view.boardW + c;
            if (t < out[k]) out[k] = t;
        };
        mark(b.row, b.col, fuse);
        for (const [dy, dx] of DIRS) {
            for (let i = 1; i <= b.power; i++) {
                const r = b.row + dy * i;
                const c = b.col + dx * i;
                if (r < 0 || r >= view.boardH || c < 0 || c >= view.boardW) break;
                const cell = view.getCell(r, c);
                if (cell === 'S') break;
                mark(r, c, fuse + i * EXPLOSION_TRAVEL_MS);
                if (cell === 'W') break;
            }
        }
    }
    return out;
}

// 4-connected BFS reachability from self. Walls (stone, wood) and bomb cells
// (other than self's current cell) block movement. Returns integer cell-depths;
// unreachable cells are Infinity. Caller multiplies by msPerCell to get ms-to-reach.
function computeReachability(view: ObsView, sr: number, sc: number): Float32Array {
    const W = view.boardW, H = view.boardH;
    const depth = new Float32Array(H * W);
    depth.fill(Infinity);
    if (sr < 0 || sr >= H || sc < 0 || sc >= W) return depth;

    const blocked = new Uint8Array(H * W);
    for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
            const cell = view.getCell(r, c);
            if (cell === 'S' || cell === 'W') blocked[r * W + c] = 1;
        }
    }
    // Bombs block, except the one (if any) on self's own cell — agent stands on it.
    for (const b of view.bombs) {
        if (b.row === sr && b.col === sc) continue;
        blocked[b.row * W + b.col] = 1;
    }

    depth[sr * W + sc] = 0;
    const queue: number[] = [sr * W + sc];
    let head = 0;
    while (head < queue.length) {
        const k = queue[head++];
        const r = (k / W) | 0;
        const c = k - r * W;
        const d = depth[k];
        for (const [dy, dx] of DIRS) {
            const nr = r + dy, nc = c + dx;
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            const nk = nr * W + nc;
            if (blocked[nk]) continue;
            if (depth[nk] <= d + 1) continue;
            depth[nk] = d + 1;
            queue.push(nk);
        }
    }
    return depth;
}

// Recover the legal-action mask from a stored observation. With egocentric
// framing self is always at (SELF_R, SELF_C); off-board cells are stone-padded
// on channel 0, so the same wall/bomb tests work at the edges. Channel layout
// contract: 0=stone, 1=wood, 2=bomb, 8=self pos, scalar[3]=placedBombs/maxBombs.
// Used at training time to mask the target argmax over Q(s_next) without storing
// masks in the PER buffer. Action order matches sim.Action: 0=STAY 1=UP 2=DOWN
// 3=LEFT 4=RIGHT 5=BOMB.
const NUM_ACTIONS_LOCAL = 6;
const MASK_DIRS: Array<[number, number, number]> = [
    [1, -1, 0], [2, 1, 0], [3, 0, -1], [4, 0, 1],
];
export function legalMaskFromObs(obs: Float32Array, offset: number = 0): Uint8Array {
    const mask = new Uint8Array(NUM_ACTIONS_LOCAL);
    mask[0] = 1; // STAY always legal
    // Self lives at the fixed center. If the channel is 0 there, agent is dead.
    if (obs[offset + idx(SELF_R, SELF_C, 8)] < 0.5) return mask;
    for (const [a, dr, dc] of MASK_DIRS) {
        const tr = SELF_R + dr;
        const tc = SELF_C + dc;
        // No bounds check needed: SELF_R/SELF_C ± 1 stays inside the view.
        if (obs[offset + idx(tr, tc, 0)] > 0.5) continue; // stone (or padding)
        if (obs[offset + idx(tr, tc, 1)] > 0.5) continue; // wood
        if (obs[offset + idx(tr, tc, 2)] > 0.5) continue; // bomb
        mask[a] = 1;
    }
    // BOMB legal if placedBombs/maxBombs<1 (scalar idx 3) and no bomb on self cell.
    if (obs[offset + OBS_SPATIAL_SIZE + 3] < 0.999 && obs[offset + idx(SELF_R, SELF_C, 2)] < 0.5) {
        mask[5] = 1;
    }
    return mask;
}

export function encode(view: ObsView): Float32Array {
    const out = new Float32Array(OBS_SIZE);
    const blastTimes = computeBlastTimes(view);

    const sr = Math.round(view.self.y);
    const sc = Math.round(view.self.x);
    // BFS is on the real board; we crop the result into the centered view below.
    const reachDepth = computeReachability(view, sr, sc);
    // Real per-cell traversal time. moveSpeed (~0.045) × SPEED_TO_CELLS_PER_MS (~0.1)
    // ≈ 0.0045 cells/ms → ~222ms per cell. Links BFS depth to the same time axis as
    // blastTimes / DANGER_LOOKAHEAD_MS.
    const cellsPerMs = view.self.moveSpeed * SPEED_TO_CELLS_PER_MS;
    const msPerCell = cellsPerMs > 0 ? 1 / cellsPerMs : Infinity;

    // Shift: encoded cell (wr, wc) reads from real cell (wr - SELF_R + sr, ...).
    for (let wr = 0; wr < BOARD_H; wr++) {
        for (let wc = 0; wc < BOARD_W; wc++) {
            const rr = wr - SELF_R + sr;
            const cc = wc - SELF_C + sc;
            if (rr < 0 || rr >= view.boardH || cc < 0 || cc >= view.boardW) {
                // Off-board → stone padding so the agent can't "see through" the edge.
                out[idx(wr, wc, 0)] = 1;
                continue;
            }
            const cell = view.getCell(rr, cc);
            if (cell === 'S') out[idx(wr, wc, 0)] = 1;
            else if (cell === 'W') out[idx(wr, wc, 1)] = 1;
            else if (cell === 'NUM') out[idx(wr, wc, 5)] = 1;
            else if (cell === 'SPE') out[idx(wr, wc, 6)] = 1;
            else if (cell === 'STR') out[idx(wr, wc, 7)] = 1;

            const bt = blastTimes[rr * view.boardW + cc];
            if (bt !== Infinity) {
                out[idx(wr, wc, 4)] = Math.max(0, 1 - bt / DANGER_LOOKAHEAD_MS);
            }

            const depth = reachDepth[rr * view.boardW + cc];
            if (depth !== Infinity) {
                const rms = depth * msPerCell;
                out[idx(wr, wc, 13)] = Math.max(0, 1 - rms / DANGER_LOOKAHEAD_MS);
                // Survivable: reach the cell strictly before it becomes lethal,
                // or it never blasts. Self's own cell (depth 0) is always reachable.
                if (bt === Infinity || rms < bt) out[idx(wr, wc, 14)] = 1;
            }
        }
    }

    // Bomb fuse/presence/power written from the bomb list so fuse values are precise.
    for (const b of view.bombs) {
        const wr = b.row - sr + SELF_R;
        const wc = b.col - sc + SELF_C;
        if (wr < 0 || wr >= BOARD_H || wc < 0 || wc >= BOARD_W) continue;
        out[idx(wr, wc, 2)] = 1;
        out[idx(wr, wc, 3)] = Math.max(0, Math.min(1, b.fuseRemainingMs / BOMB_FUSE_MS));
        out[idx(wr, wc, 12)] = Math.max(0, Math.min(1, b.power / MAX_POWERUP_TIER));
    }

    // Self is fixed at the view center (only emitted if actually alive/on-board).
    if (sr >= 0 && sr < view.boardH && sc >= 0 && sc < view.boardW) {
        out[idx(SELF_R, SELF_C, 8)] = 1;
    }

    for (const e of view.enemies) {
        if (!e.alive) continue;
        const er = Math.round(e.y);
        const ec = Math.round(e.x);
        if (er < 0 || er >= view.boardH || ec < 0 || ec >= view.boardW) continue;
        const wr = er - sr + SELF_R;
        const wc = ec - sc + SELF_C;
        if (wr < 0 || wr >= BOARD_H || wc < 0 || wc >= BOARD_W) continue;
        if (e.knocked) {
            out[idx(wr, wc, 10)] = Math.max(0, Math.min(1, e.knockMsLeft / KNOCK_DURATION_MS));
        } else {
            out[idx(wr, wc, 9)] = 1;
        }
    }

    for (const ab of view.activeBlasts) {
        if (ab.row < 0 || ab.row >= view.boardH || ab.col < 0 || ab.col >= view.boardW) continue;
        const wr = ab.row - sr + SELF_R;
        const wc = ab.col - sc + SELF_C;
        if (wr < 0 || wr >= BOARD_H || wc < 0 || wc >= BOARD_W) continue;
        const v = Math.max(0, Math.min(1, ab.msUntilSafe / ACTIVE_BLAST_DANGER_MS));
        // Multiple overlapping explosions on the same cell → keep the larger danger.
        const cur = out[idx(wr, wc, 11)];
        if (v > cur) out[idx(wr, wc, 11)] = v;
    }

    const off = OBS_SPATIAL_SIZE;
    out[off + 0] = view.self.bombPower / MAX_POWERUP_TIER;
    out[off + 1] = view.self.maxBombs / MAX_POWERUP_TIER;
    out[off + 2] = (Math.round((view.self.moveSpeed - START_MOVE_SPEED) / MOVE_SPEED_INCREMENT) + 1) / MAX_POWERUP_TIER;
    out[off + 3] = view.self.maxBombs > 0 ? view.self.placedBombs / view.self.maxBombs : 0;
    out[off + 4] = view.woodLeft / 100;
    out[off + 5] = 1 - Math.min(1, view.woodLeft / 40);

    // Safe-neighbors freedom signal: count of 4-connected neighbors of the real
    // self cell that are BFS-reachable AND survivable. Done in real-board coords
    // (not the cropped view) so it's stable regardless of self position.
    let safeCount = 0;
    if (sr >= 0 && sr < view.boardH && sc >= 0 && sc < view.boardW) {
        for (const [dy, dx] of DIRS) {
            const nr = sr + dy, nc = sc + dx;
            if (nr < 0 || nr >= view.boardH || nc < 0 || nc >= view.boardW) continue;
            const k = nr * view.boardW + nc;
            const depth = reachDepth[k];
            if (depth === Infinity) continue;
            const rms = depth * msPerCell;
            const bt = blastTimes[k];
            if (bt === Infinity || rms < bt) safeCount++;
        }
    }
    out[off + 6] = safeCount / 4;

    return out;
}
