// Observation encoder — shared between sim-side training and gameserver-side inference.
// We deliberately keep it framework-free (plain Float32Array) so bot2.ts can call it
// with no TF.js dependency.
//
// Layout: spatial portion is BOARD_H × BOARD_W × NUM_CHANNELS flattened in NHWC order
// (channels-last) so the flat buffer can be reshaped directly into a TF.js conv2d
// input tensor. Scalar features are appended after. The encoder is egocentric: the
// agent's own self-channel and stats are placed first, followed by enemy channels in
// fixed order.

export const BOARD_H = 13;
export const BOARD_W = 19;

// Spatial channels:
//   0  stone (immovable)
//   1  wood (destroyable)
//   2  bomb present
//   3  bomb fuse remaining (1.0 = just placed, 0.0 = about to explode)
//   4  time-to-blast at each cell (1.0 = exploding now, 0.0 = >=DANGER_LOOKAHEAD_MS away)
//   5  powerup NUM
//   6  powerup SPE
//   7  powerup STR
//   8  self position
//   9  enemy positions (any alive non-knocked enemy)
//   10 knocked enemy positions (vulnerable — easy kill)
export const NUM_CHANNELS = 11;
// Scalar features (normalized):
//   0  my bombPower / 13
//   1  my maxBombs / 13
//   2  my moveSpeed tier / 13
//   3  my placedBombs / maxBombs
//   4  wood blocks remaining / 100
//   5  game phase (0 = lots of walls, 1 = none)
export const NUM_SCALARS = 6;

export const OBS_SPATIAL_SIZE = BOARD_H * BOARD_W * NUM_CHANNELS;
export const OBS_SIZE = OBS_SPATIAL_SIZE + NUM_SCALARS;

const BOMB_FUSE_MS = 3000;
const DANGER_LOOKAHEAD_MS = 1500;
const MAX_POWERUP_TIER = 13;
const START_MOVE_SPEED = 0.045;
const MOVE_SPEED_INCREMENT = 0.003;
// Matches sim.ts EXPLOSION_TRAVEL_MS and production game.ts:12. The far end of a
// power=10 bomb is +250ms past the center — material to the agent's dodge plan.
const EXPLOSION_TRAVEL_MS = 25;

// A minimal interface so this module doesn't depend on Sim itself — bot2.ts builds its
// own SimView from received network state.
export interface ObsView {
    boardH: number;
    boardW: number;
    // 'S' | 'W' | 'B' | 'NUM' | 'SPE' | 'STR' | 'E'
    getCell(r: number, c: number): string;
    // Bombs on the board with their remaining fuse in ms and power.
    bombs: Array<{ row: number; col: number; power: number; fuseRemainingMs: number }>;
    self: { y: number; x: number; bombPower: number; maxBombs: number; moveSpeed: number; placedBombs: number };
    enemies: Array<{ y: number; x: number; alive: boolean; knocked: boolean }>;
    woodLeft: number;
}

// NHWC (channels-last) flat index: [r, x, c] → r*W*C + x*C + c.
function idx(r: number, x: number, c: number): number {
    return r * BOARD_W * NUM_CHANNELS + x * NUM_CHANNELS + c;
}

// Compute, for each cell, the minimum ms until it gets blasted by some pending bomb.
// Distance-i along a ray gets hit at fuseRemainingMs + i * 25ms — matches sim.ts and
// production game.ts/executeExplosions. Chain reactions only ever LOWER these timings,
// so this is a safe lower bound for the agent's dodge planning.
function computeBlastTimes(view: ObsView): Float32Array {
    const out = new Float32Array(view.boardH * view.boardW);
    out.fill(Infinity);
    for (const b of view.bombs) {
        const mark = (r: number, c: number, t: number) => {
            const k = r * view.boardW + c;
            if (t < out[k]) out[k] = t;
        };
        mark(b.row, b.col, b.fuseRemainingMs);
        const tryDir = (dy: number, dx: number) => {
            for (let i = 1; i <= b.power; i++) {
                const r = b.row + dy * i;
                const c = b.col + dx * i;
                if (r < 0 || r >= view.boardH || c < 0 || c >= view.boardW) break;
                const cell = view.getCell(r, c);
                if (cell === 'S') break;
                mark(r, c, b.fuseRemainingMs + i * EXPLOSION_TRAVEL_MS);
                if (cell === 'W') break; // wood absorbs the ray
            }
        };
        tryDir(-1, 0); tryDir(1, 0); tryDir(0, -1); tryDir(0, 1);
    }
    return out;
}

export function encode(view: ObsView): Float32Array {
    const out = new Float32Array(OBS_SIZE);
    const blastTimes = computeBlastTimes(view);

    for (let r = 0; r < view.boardH; r++) {
        for (let c = 0; c < view.boardW; c++) {
            const cell = view.getCell(r, c);
            if (cell === 'S') out[idx(r, c, 0)] = 1;
            else if (cell === 'W') out[idx(r, c, 1)] = 1;
            else if (cell === 'NUM') out[idx(r, c, 5)] = 1;
            else if (cell === 'SPE') out[idx(r, c, 6)] = 1;
            else if (cell === 'STR') out[idx(r, c, 7)] = 1;

            // Bomb-specific channels look up the actual bomb, not just 'B' presence.
            // Done below in the bomb loop for precise fuse values.
        }
    }
    for (const b of view.bombs) {
        out[idx(b.row, b.col, 2)] = 1;
        // Higher value = more time left = less urgent.
        out[idx(b.row, b.col, 3)] = Math.max(0, Math.min(1, b.fuseRemainingMs / BOMB_FUSE_MS));
    }
    // Blast danger: closer-to-detonation = higher.
    for (let r = 0; r < view.boardH; r++) {
        for (let c = 0; c < view.boardW; c++) {
            const t = blastTimes[r * view.boardW + c];
            if (t === Infinity) continue;
            const v = Math.max(0, 1 - t / DANGER_LOOKAHEAD_MS);
            out[idx(r, c, 4)] = v;
        }
    }
    // Self position (egocentric: only my channel lights up here).
    const sr = Math.round(view.self.y);
    const sx = Math.round(view.self.x);
    if (sr >= 0 && sr < view.boardH && sx >= 0 && sx < view.boardW) out[idx(sr, sx, 8)] = 1;

    for (const e of view.enemies) {
        if (!e.alive) continue;
        const er = Math.round(e.y);
        const ex = Math.round(e.x);
        if (er < 0 || er >= view.boardH || ex < 0 || ex >= view.boardW) continue;
        if (e.knocked) out[idx(er, ex, 10)] = 1;
        else out[idx(er, ex, 9)] = 1;
    }

    // Scalars (appended after the spatial features).
    const off = OBS_SPATIAL_SIZE;
    out[off + 0] = view.self.bombPower / MAX_POWERUP_TIER;
    out[off + 1] = view.self.maxBombs / MAX_POWERUP_TIER;
    out[off + 2] = (Math.round((view.self.moveSpeed - START_MOVE_SPEED) / MOVE_SPEED_INCREMENT) + 1) / MAX_POWERUP_TIER;
    out[off + 3] = view.self.maxBombs > 0 ? view.self.placedBombs / view.self.maxBombs : 0;
    out[off + 4] = view.woodLeft / 100;
    out[off + 5] = 1 - Math.min(1, view.woodLeft / 40);

    return out;
}
