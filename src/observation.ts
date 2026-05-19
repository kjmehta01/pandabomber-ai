// Observation encoder shared by sim-side training and gameserver-side inference.
// Framework-free (plain Float32Array) so bot2.ts can call it without TF.js.
//
// Spatial layout is BOARD_H × BOARD_W × NUM_CHANNELS in NHWC order so the buffer
// reshapes directly into a conv2d tensor. Scalars are appended after. Egocentric:
// the self-channel comes before enemy channels.

export const BOARD_H = 13;
export const BOARD_W = 19;

// Spatial channels:
//   0  stone               1  wood                2  bomb present
//   3  bomb fuse (1=just placed, 0=about to explode)
//   4  time-to-blast (1=exploding now, 0=>=DANGER_LOOKAHEAD_MS away)
//   5  powerup NUM         6  powerup SPE         7  powerup STR
//   8  self position       9  enemy positions
//  10  knocked enemies: knockMsLeft/KNOCK_DURATION_MS (1=just knocked, 0=recovering now)
export const NUM_CHANNELS = 11;
// Scalars (all normalized): bombPower/13, maxBombs/13, moveSpeed tier/13,
// placedBombs/maxBombs, woodLeft/100, game phase (0=lots of walls, 1=none).
export const NUM_SCALARS = 6;

export const OBS_SPATIAL_SIZE = BOARD_H * BOARD_W * NUM_CHANNELS;
export const OBS_SIZE = OBS_SPATIAL_SIZE + NUM_SCALARS;

const BOMB_FUSE_MS = 3000;
// Covers the full fuse so the danger map lights up at placement time.
const DANGER_LOOKAHEAD_MS = 3000;
const KNOCK_DURATION_MS = 6000;
const MAX_POWERUP_TIER = 13;
const START_MOVE_SPEED = 0.045;
const MOVE_SPEED_INCREMENT = 0.003;
// Matches sim.ts and production game.ts:12. A power=10 bomb's far end is +250ms
// past center — material to the agent's dodge plan.
const EXPLOSION_TRAVEL_MS = 25;

// Minimal interface so this module doesn't depend on Sim; bot2.ts builds its own
// SimView from network state.
export interface ObsView {
    boardH: number;
    boardW: number;
    getCell(r: number, c: number): string;
    bombs: Array<{ row: number; col: number; power: number; fuseRemainingMs: number }>;
    self: { y: number; x: number; bombPower: number; maxBombs: number; moveSpeed: number; placedBombs: number };
    enemies: Array<{ y: number; x: number; alive: boolean; knocked: boolean; knockMsLeft: number }>;
    woodLeft: number;
}

// NHWC flat index.
function idx(r: number, x: number, c: number): number {
    return r * BOARD_W * NUM_CHANNELS + x * NUM_CHANNELS + c;
}

// Min ms until each cell gets blasted by some pending bomb. Distance-i along a ray
// arrives at fuseRemainingMs + i*25ms. Chain reactions can only LOWER this, so the
// value is a safe lower bound for dodge planning.
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
                if (cell === 'W') break;
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
        }
    }
    // Bomb channels written below from the bomb list so fuse values are precise.
    for (const b of view.bombs) {
        out[idx(b.row, b.col, 2)] = 1;
        out[idx(b.row, b.col, 3)] = Math.max(0, Math.min(1, b.fuseRemainingMs / BOMB_FUSE_MS));
    }
    for (let r = 0; r < view.boardH; r++) {
        for (let c = 0; c < view.boardW; c++) {
            const t = blastTimes[r * view.boardW + c];
            if (t === Infinity) continue;
            const v = Math.max(0, 1 - t / DANGER_LOOKAHEAD_MS);
            out[idx(r, c, 4)] = v;
        }
    }
    const sr = Math.round(view.self.y);
    const sx = Math.round(view.self.x);
    if (sr >= 0 && sr < view.boardH && sx >= 0 && sx < view.boardW) out[idx(sr, sx, 8)] = 1;

    for (const e of view.enemies) {
        if (!e.alive) continue;
        const er = Math.round(e.y);
        const ex = Math.round(e.x);
        if (er < 0 || er >= view.boardH || ex < 0 || ex >= view.boardW) continue;
        if (e.knocked) {
            out[idx(er, ex, 10)] = Math.max(0, Math.min(1, e.knockMsLeft / KNOCK_DURATION_MS));
        } else {
            out[idx(er, ex, 9)] = 1;
        }
    }

    const off = OBS_SPATIAL_SIZE;
    out[off + 0] = view.self.bombPower / MAX_POWERUP_TIER;
    out[off + 1] = view.self.maxBombs / MAX_POWERUP_TIER;
    out[off + 2] = (Math.round((view.self.moveSpeed - START_MOVE_SPEED) / MOVE_SPEED_INCREMENT) + 1) / MAX_POWERUP_TIER;
    out[off + 3] = view.self.maxBombs > 0 ? view.self.placedBombs / view.self.maxBombs : 0;
    out[off + 4] = view.woodLeft / 100;
    out[off + 5] = 1 - Math.min(1, view.woodLeft / 40);

    return out;
}
