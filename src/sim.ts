// Pure-logic simulator for PandaBomber. Tracks production game.ts/gameplayer.ts as
// closely as practical at a 10ms tick — matches production's 100Hz tickRate exactly.
// The non-trivial timing pieces — explosion travel and per-cell kill windows — are
// modeled explicitly; see ActiveExplosion.
//
// Sim is MU-only. Time-trial (TT) game mechanics from production (deterministic
// powerup rotation, wood-clearing win condition) are not modeled — the trained bot
// is only ever used in 4-player MU matches.
//
// Exactly modeled:
//   - bomb fuse (3000ms), invulnerability (200ms), knock duration (6000ms)
//   - 3-strikes-same-cell ⇒ outright death (gameplayer.ts:70-84)
//   - chain detonation: ray hitting a bomb chains synchronously, ray PASSES THROUGH
//     the bomb's cell (production explodeBomb: no break after recursive chain call)
//   - rays stop at stone / wood; wood absorbs the ray and is destroyed AT THE RAY'S
//     ARRIVAL TIME (production recurseExecute), not at fuse-expiry
//   - 25ms-per-cell explosion travel; per-cell death window = [arrival, arrival+50ms]
//     (matches production's two checkPlayerDeaths calls per cell)
//   - powerup caps (13), MU spawn odds, death-drop spawn (4 in 5×5 around death)
//   - corner spawn clearing (3 cells per corner are wood-free)
//
// Movement passability uses Math.round() of the target cell — exactly what the
// authoritative server check does (gameplayer.ts:48-60 updatePos). The frontend's
// ceil/floor logic with bombWalkingTolerance is client-side prediction for smooth
// rendering of human input; it doesn't affect what the server accepts.
//
// Kill checks fire exactly twice per cell: once at first tick `now >= arrivalMs`,
// once at first tick `now >= arrivalMs + 50ms`. After that the cell is permanently
// safe (production: the two setTimeouts in recurseExecute have fired and there are
// no more pending checks). Each "first tick past T" may be up to SIM_DT_MS=10ms late
// vs production's exact-timer firing — that's our only timing-quantization error.

export const BOARD_H = 13;
export const BOARD_W = 19;

export const SIM_DT_MS = 10;
export const BOMB_FUSE_MS = 3000;
export const KNOCK_DURATION_MS = 6000;
export const INVULNERABILITY_MS = 200;
export const MAX_POWERUPS = 13;
export const EXPLOSION_TRAVEL_MS = 25; // matches game.ts:12
export const DEATH_CHECK_WINDOW_MS = 50; // matches the second checkPlayerDeaths +50ms in executeExplosions/recurseExecute

const START_MOVE_SPEED = 0.045;
const MOVE_SPEED_INCREMENT = 0.003;
// Production: cells/frame = moveSpeed * 1.667; frames at 60Hz ⇒ cells/sec = moveSpeed*1.667*60.
const CELLS_PER_MS = (speed: number) => speed * 1.667 * 60 / 1000;

export type Cell = 'E' | 'S' | 'W' | 'B' | 'NUM' | 'SPE' | 'STR';
export type Action = 0 | 1 | 2 | 3 | 4 | 5;
export const ACTION_STAY = 0;
export const ACTION_UP = 1;
export const ACTION_DOWN = 2;
export const ACTION_LEFT = 3;
export const ACTION_RIGHT = 4;
export const ACTION_BOMB = 5;
export const NUM_ACTIONS = 6;

export interface Bomb {
    row: number;
    col: number;
    power: number;
    fuseRemainingMs: number;
    ownerIdx: number;
}

export interface SimPlayer {
    idx: number;
    alive: boolean;
    knocked: boolean;
    immuneMs: number;
    knockMsLeft: number;
    y: number;
    x: number;
    // Sign of motion (-1, 0, 1). Magnitude comes from moveSpeed via CELLS_PER_MS.
    dyDir: number;
    dxDir: number;
    // Cell we're actively traveling toward; set at commit, cleared on arrival.
    moveTargetY: number;
    moveTargetX: number;
    pendingAction: Action;
    moveSpeed: number;
    bombPower: number;
    maxBombs: number;
    placedBombs: number;
    lastDeathRow: number;
    lastDeathCol: number;
    numDies: number;
    firstDieTimeMs: number;
    rewardThisStep: number;
    // Per-game counters surfaced by the eval harness. Not used during training, but
    // updates are O(1) per event so the cost is negligible.
    stats: {
        bombsPlaced: number;
        woodDestroyed: number;
        powerupsCollected: number;
        killsScored: number;
        diedFromOwnBomb: number;
    };
}

// One active explosion event. `cells` keys are "r,c" strings; each entry tracks when
// the ray arrives and when its second kill check fires. We model production exactly:
// two kill checks per cell, at arrival and arrival+50ms, then never again.
//
// If multiple rays reach the same cell (e.g. chains), we keep the EARLIEST arrival —
// the first ray's two checks dominate the cell's danger window, and the production
// kill-on-knock/200ms-invulnerability rule absorbs any extra firings from later rays.
export interface ExplosionCell {
    arrivalMs: number;
    secondCheckMs: number; // arrivalMs + DEATH_CHECK_WINDOW_MS
    sourceOwnerIdx: number;
    firstFired: boolean;
    secondFired: boolean;
}

export interface ActiveExplosion {
    cells: Map<string, ExplosionCell>;
}

const DIR_DY = [0, -1, 1, 0, 0, 0];
const DIR_DX = [0, 0, 0, -1, 1, 0];
const DIRS: Array<[number, number]> = [[-1, 0], [1, 0], [0, -1], [0, 1]];

export interface SimConfig {
    numPlayers: number;
    woodOdds?: number;
    powerupOdds?: number;
    seed?: number;
    maxTimeMs?: number;
}

function makeRng(seed: number) {
    let s = seed | 0 || 1;
    return () => {
        s ^= s << 13;
        s ^= s >>> 17;
        s ^= s << 5;
        return ((s >>> 0) / 0x100000000);
    };
}

export class Sim {
    cfg: Required<SimConfig>;
    blocks: (Cell | undefined)[][];
    bombs: (Bomb | undefined)[][];
    powerups: ('NUM' | 'SPE' | 'STR' | undefined)[][];
    players: SimPlayer[];
    activeExplosions: ActiveExplosion[];
    woodLeft: number;
    elapsedMs: number;
    ranking: number[];
    done: boolean;
    rng: () => number;

    constructor(cfg: SimConfig) {
        this.cfg = {
            numPlayers: cfg.numPlayers,
            woodOdds: cfg.woodOdds ?? 0.8,
            powerupOdds: cfg.powerupOdds ?? 0.4,
            seed: cfg.seed ?? Math.floor(Math.random() * 1e9),
            maxTimeMs: cfg.maxTimeMs ?? 180_000,
        };
        this.rng = makeRng(this.cfg.seed);

        this.blocks = [];
        this.bombs = [];
        this.powerups = [];
        this.activeExplosions = [];
        for (let r = 0; r < BOARD_H; r++) {
            this.blocks.push(new Array(BOARD_W).fill(undefined));
            this.bombs.push(new Array(BOARD_W).fill(undefined));
            this.powerups.push(new Array(BOARD_W).fill(undefined));
        }

        for (let r = 0; r < BOARD_H; r++) {
            for (let c = 0; c < BOARD_W; c++) {
                if (r === 0 || r === BOARD_H - 1 || c === 0 || c === BOARD_W - 1) {
                    this.blocks[r][c] = 'S';
                } else if (r % 2 === 0 && c % 2 === 0) {
                    this.blocks[r][c] = 'S';
                }
            }
        }

        const corners: [number, number][][] = [
            [[1, 1], [1, 2], [2, 1]],
            [[1, BOARD_W - 2], [1, BOARD_W - 3], [2, BOARD_W - 2]],
            [[BOARD_H - 2, 1], [BOARD_H - 2, 2], [BOARD_H - 3, 1]],
            [[BOARD_H - 2, BOARD_W - 2], [BOARD_H - 2, BOARD_W - 3], [BOARD_H - 3, BOARD_W - 2]],
        ];
        const cornerSet = new Set<string>();
        for (const c of corners) for (const [r, x] of c) cornerSet.add(`${r},${x}`);

        this.woodLeft = 0;
        for (let r = 1; r < BOARD_H - 1; r++) {
            for (let c = 1; c < BOARD_W - 1; c++) {
                if (this.blocks[r][c]) continue;
                if (cornerSet.has(`${r},${c}`)) continue;
                if (this.rng() < this.cfg.woodOdds) {
                    this.blocks[r][c] = 'W';
                    this.woodLeft++;
                }
            }
        }

        const corner: [number, number][] = [
            [1, 1],
            [1, BOARD_W - 2],
            [BOARD_H - 2, 1],
            [BOARD_H - 2, BOARD_W - 2],
        ];
        const order = [0, 1, 2, 3];
        for (let i = order.length - 1; i > 0; i--) {
            const j = Math.floor(this.rng() * (i + 1));
            [order[i], order[j]] = [order[j], order[i]];
        }
        this.players = [];
        for (let i = 0; i < this.cfg.numPlayers; i++) {
            const [py, px] = corner[order[i]];
            this.players.push({
                idx: i, alive: true, knocked: false, immuneMs: 0, knockMsLeft: 0,
                y: py, x: px, dyDir: 0, dxDir: 0, moveTargetY: py, moveTargetX: px,
                pendingAction: ACTION_STAY,
                moveSpeed: START_MOVE_SPEED, bombPower: 1, maxBombs: 1, placedBombs: 0,
                lastDeathRow: -1, lastDeathCol: -1, numDies: 0, firstDieTimeMs: -1,
                rewardThisStep: 0,
                stats: { bombsPlaced: 0, woodDestroyed: 0, powerupsCollected: 0, killsScored: 0, diedFromOwnBomb: 0 },
            });
        }
        this.elapsedMs = 0;
        this.ranking = [];
        this.done = false;
    }

    getCell(r: number, c: number): Cell {
        if (r < 0 || r >= BOARD_H || c < 0 || c >= BOARD_W) return 'S';
        if (this.blocks[r][c] === 'S') return 'S';
        if (this.blocks[r][c] === 'W') return 'W';
        if (this.bombs[r][c]) return 'B';
        if (this.powerups[r][c]) return this.powerups[r][c]!;
        return 'E';
    }

    private isPassable(r: number, c: number): boolean {
        const cell = this.getCell(r, c);
        return cell === 'E' || cell === 'B' || cell === 'NUM' || cell === 'SPE' || cell === 'STR';
    }

    // True iff the player is at integer cell coords (within tolerance) — both the
    // commit point AND the only point at which a new action can take effect.
    isAtCell(playerIdx: number): boolean {
        const p = this.players[playerIdx];
        return p.dyDir === 0 && p.dxDir === 0;
    }

    setAction(playerIdx: number, action: Action) {
        const p = this.players[playerIdx];
        if (!p.alive) return;
        p.pendingAction = action;
    }

    private commitActions() {
        for (const p of this.players) {
            if (!p.alive || p.knocked) continue;
            if (p.dyDir !== 0 || p.dxDir !== 0) continue; // mid-cell; can't commit
            const a = p.pendingAction;
            p.pendingAction = ACTION_STAY; // consumed
            if (a === ACTION_STAY) continue;
            if (a === ACTION_BOMB) {
                const r = Math.round(p.y);
                const c = Math.round(p.x);
                this.placeBomb(p, r, c);
                continue;
            }
            const dy = DIR_DY[a];
            const dx = DIR_DX[a];
            const tr = Math.round(p.y) + dy;
            const tc = Math.round(p.x) + dx;
            if (!this.isPassable(tr, tc)) continue;
            p.moveTargetY = tr;
            p.moveTargetX = tc;
            p.dyDir = dy;
            p.dxDir = dx;
        }
    }

    private placeBomb(p: SimPlayer, r: number, c: number) {
        if (p.placedBombs >= p.maxBombs) return;
        if (this.bombs[r][c]) return;
        if (this.blocks[r][c]) return;
        this.bombs[r][c] = { row: r, col: c, power: p.bombPower, fuseRemainingMs: BOMB_FUSE_MS, ownerIdx: p.idx };
        p.placedBombs++;
        p.stats.bombsPlaced++;
    }

    private movePlayers(dtMs: number) {
        for (const p of this.players) {
            if (!p.alive || p.knocked) continue;
            if (p.dyDir === 0 && p.dxDir === 0) continue;
            const stepCells = CELLS_PER_MS(p.moveSpeed) * dtMs;
            if (p.dyDir !== 0) {
                const remaining = p.moveTargetY - p.y;
                if (Math.abs(remaining) <= stepCells) {
                    p.y = p.moveTargetY;
                    p.dyDir = 0;
                } else {
                    p.y += p.dyDir * stepCells;
                }
            } else if (p.dxDir !== 0) {
                const remaining = p.moveTargetX - p.x;
                if (Math.abs(remaining) <= stepCells) {
                    p.x = p.moveTargetX;
                    p.dxDir = 0;
                } else {
                    p.x += p.dxDir * stepCells;
                }
            }
        }
    }

    /* ─── explosion machinery ────────────────────────────────────────── */

    private advanceBombs(dtMs: number) {
        for (let r = 0; r < BOARD_H; r++) {
            for (let c = 0; c < BOARD_W; c++) {
                const b = this.bombs[r][c];
                if (b) b.fuseRemainingMs -= dtMs;
            }
        }
        const processed = new Set<Bomb>();
        for (let r = 0; r < BOARD_H; r++) {
            for (let c = 0; c < BOARD_W; c++) {
                const b = this.bombs[r][c];
                if (!b) continue;
                if (b.fuseRemainingMs > 0) continue;
                if (processed.has(b)) continue;
                const chain = this.buildChain(b);
                for (const cb of chain) processed.add(cb);
                this.createExplosion(chain);
            }
        }
    }

    // Find all bombs that detonate together. Mirrors production explodeBomb's recursive
    // chain: a ray that hits a bomb chains it (no break), but wood/stone stop the ray
    // entirely, so they also stop the chain through that direction.
    private buildChain(root: Bomb): Bomb[] {
        const chain = new Set<Bomb>();
        chain.add(root);
        const queue: Bomb[] = [root];
        while (queue.length > 0) {
            const b = queue.shift()!;
            for (const [dy, dx] of DIRS) {
                for (let i = 1; i <= b.power; i++) {
                    const r = b.row + dy * i;
                    const c = b.col + dx * i;
                    if (r < 0 || r >= BOARD_H || c < 0 || c >= BOARD_W) break;
                    if (this.blocks[r][c] === 'S') break;
                    if (this.blocks[r][c] === 'W') break; // wood blocks the chain
                    const other = this.bombs[r][c];
                    if (other && !chain.has(other)) {
                        chain.add(other);
                        queue.push(other);
                    }
                    // ray continues past empties / powerups / chained bombs
                }
            }
        }
        return [...chain];
    }

    // Compute per-cell arrival times across all centers, mark the resulting cells as
    // an ActiveExplosion. Wood destruction and player damage are NOT done here — they
    // happen in processActiveExplosions at the cell's arrival tick.
    private createExplosion(centers: Bomb[]) {
        const cells = new Map<string, ExplosionCell>();
        const explodeAt = this.elapsedMs;

        const upsert = (r: number, c: number, arrivalT: number, owner: number) => {
            const key = `${r},${c}`;
            const ex = cells.get(key);
            if (!ex) {
                cells.set(key, {
                    arrivalMs: arrivalT,
                    secondCheckMs: arrivalT + DEATH_CHECK_WINDOW_MS,
                    sourceOwnerIdx: owner,
                    firstFired: false,
                    secondFired: false,
                });
            } else if (arrivalT < ex.arrivalMs) {
                // Earlier ray wins: it owns the wood-destroy credit and bounds the
                // first kill check.
                ex.arrivalMs = arrivalT;
                ex.secondCheckMs = arrivalT + DEATH_CHECK_WINDOW_MS;
                ex.sourceOwnerIdx = owner;
            }
        };

        for (const b of centers) {
            upsert(b.row, b.col, explodeAt, b.ownerIdx);
            for (const [dy, dx] of DIRS) {
                for (let i = 1; i <= b.power; i++) {
                    const r = b.row + dy * i;
                    const c = b.col + dx * i;
                    if (r < 0 || r >= BOARD_H || c < 0 || c >= BOARD_W) break;
                    if (this.blocks[r][c] === 'S') break;
                    const arrivalT = explodeAt + i * EXPLOSION_TRAVEL_MS;
                    upsert(r, c, arrivalT, b.ownerIdx);
                    if (this.blocks[r][c] === 'W') break; // wood absorbs and ends ray
                }
            }
        }

        for (const b of centers) {
            this.bombs[b.row][b.col] = undefined;
            this.players[b.ownerIdx].placedBombs--;
        }

        this.activeExplosions.push({ cells });
    }

    private processActiveExplosions() {
        const now = this.elapsedMs;
        const stillActive: ActiveExplosion[] = [];
        for (const exp of this.activeExplosions) {
            let anyUnfired = false;
            for (const [key, cell] of exp.cells) {
                if (cell.firstFired && cell.secondFired) continue;
                const [r, c] = key.split(',').map(Number);

                // First check: destroy wood + kill check at arrival (or first tick past it).
                if (!cell.firstFired && now >= cell.arrivalMs) {
                    cell.firstFired = true;
                    if (this.blocks[r][c] === 'W') {
                        this.blocks[r][c] = undefined;
                        this.woodLeft--;
                        this.players[cell.sourceOwnerIdx].rewardThisStep += 0.2;
                        this.players[cell.sourceOwnerIdx].stats.woodDestroyed++;
                        this.maybeSpawnPowerup(r, c);
                    }
                    this.killCheck(r, c, cell.sourceOwnerIdx);
                }

                // Second check: kill check at arrival+50ms (or first tick past it).
                if (cell.firstFired && !cell.secondFired && now >= cell.secondCheckMs) {
                    cell.secondFired = true;
                    this.killCheck(r, c, cell.sourceOwnerIdx);
                }

                if (!cell.firstFired || !cell.secondFired) anyUnfired = true;
            }
            // Prune once every cell has fired both checks — production stops checking
            // a cell forever after its +50ms timeout fires.
            if (anyUnfired) stillActive.push(exp);
        }
        this.activeExplosions = stillActive;
    }

    private killCheck(r: number, c: number, attackerIdx: number) {
        for (const p of this.players) {
            if (!p.alive) continue;
            if (Math.round(p.y) === r && Math.round(p.x) === c) {
                this.damagePlayer(p, this.players[attackerIdx]);
            }
        }
    }

    /* ─── player damage / death ──────────────────────────────────────── */

    private damagePlayer(p: SimPlayer, attacker: SimPlayer) {
        if (p.immuneMs > 0) return;
        p.immuneMs = INVULNERABILITY_MS;

        const row = Math.round(p.y);
        const col = Math.round(p.x);
        if (!p.knocked) {
            p.knocked = true;
            p.knockMsLeft = KNOCK_DURATION_MS;
            p.dyDir = 0; p.dxDir = 0; // freeze in place per production
            if (row === p.lastDeathRow && col === p.lastDeathCol && this.elapsedMs - p.firstDieTimeMs < 19_000) {
                if (p.numDies === 2) {
                    this.kill(p, attacker);
                    return;
                }
                p.numDies++;
            } else {
                p.lastDeathRow = row;
                p.lastDeathCol = col;
                p.numDies = 1;
                p.firstDieTimeMs = this.elapsedMs;
            }
            if (attacker !== p) attacker.rewardThisStep += 5.0;
            p.rewardThisStep -= 10.0;
        } else {
            this.kill(p, attacker);
        }
    }

    private kill(p: SimPlayer, attacker: SimPlayer) {
        p.alive = false;
        p.knocked = false;
        p.dyDir = 0; p.dxDir = 0;
        if (!this.ranking.includes(p.idx)) this.ranking.unshift(p.idx);
        if (attacker !== p) {
            attacker.rewardThisStep += 15.0;
            attacker.stats.killsScored++;
        } else {
            p.stats.diedFromOwnBomb = 1;
        }
        p.rewardThisStep -= 30.0;

        const numDrop = 4;
        const spots: [number, number][] = [];
        for (let dy = -2; dy <= 2; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
                const r = Math.round(p.y) + dy;
                const c = Math.round(p.x) + dx;
                if (r > 0 && r < BOARD_H - 1 && c > 0 && c < BOARD_W - 1 && this.getCell(r, c) === 'E') {
                    spots.push([r, c]);
                }
            }
        }
        for (let i = spots.length - 1; i > 0; i--) {
            const j = Math.floor(this.rng() * (i + 1));
            [spots[i], spots[j]] = [spots[j], spots[i]];
        }
        for (let i = 0; i < Math.min(numDrop, spots.length); i++) {
            const [r, c] = spots[i];
            this.placePowerupRandom(r, c);
        }
    }

    private maybeSpawnPowerup(r: number, c: number) {
        const roll = this.rng();
        const odds = this.cfg.powerupOdds;
        if (roll < odds / 3) this.powerups[r][c] = 'NUM';
        else if (roll < (odds * 2) / 3) this.powerups[r][c] = 'SPE';
        else if (roll < odds) this.powerups[r][c] = 'STR';
    }

    private placePowerupRandom(r: number, c: number) {
        const roll = this.rng();
        if (roll < 1 / 3) this.powerups[r][c] = 'NUM';
        else if (roll < 2 / 3) this.powerups[r][c] = 'SPE';
        else this.powerups[r][c] = 'STR';
    }

    private collectPowerups() {
        for (const p of this.players) {
            if (!p.alive) continue;
            const r = Math.round(p.y);
            const c = Math.round(p.x);
            const pup = this.powerups[r][c];
            if (!pup) continue;
            this.powerups[r][c] = undefined;
            if (pup === 'NUM' && p.maxBombs < MAX_POWERUPS) p.maxBombs++;
            else if (pup === 'STR' && p.bombPower < MAX_POWERUPS) p.bombPower++;
            else if (pup === 'SPE') {
                const tier = Math.round((p.moveSpeed - START_MOVE_SPEED) / MOVE_SPEED_INCREMENT) + 1;
                if (tier < MAX_POWERUPS) p.moveSpeed += MOVE_SPEED_INCREMENT;
            }
            p.rewardThisStep += 0.5;
            p.stats.powerupsCollected++;
        }
    }

    private updateTimers(dtMs: number) {
        for (const p of this.players) {
            if (p.immuneMs > 0) p.immuneMs = Math.max(0, p.immuneMs - dtMs);
            if (p.knocked) {
                p.knockMsLeft -= dtMs;
                if (p.knockMsLeft <= 0) {
                    p.knocked = false;
                    p.knockMsLeft = 0;
                }
            }
        }
    }

    private gameOver(): boolean {
        const alive = this.players.filter(p => p.alive);
        if (alive.length === 0) return true;
        if (alive.length === 1 && this.cfg.numPlayers > 1) {
            if (!this.ranking.includes(alive[0].idx)) this.ranking.unshift(alive[0].idx);
            alive[0].rewardThisStep += 30.0;
            return true;
        }
        if (this.elapsedMs >= this.cfg.maxTimeMs) {
            // Stalemate: penalize every player still alive at timeout. Without this,
            // mutual avoidance is a viable strategy — the per-step penalty alone
            // doesn't deter two scared bots from both surviving to the clock.
            for (const p of this.players) {
                if (p.alive) {
                    p.rewardThisStep -= 15.0;
                    if (!this.ranking.includes(p.idx)) this.ranking.unshift(p.idx);
                }
            }
            return true;
        }
        return false;
    }

    step(): { rewards: number[]; done: boolean } {
        for (const p of this.players) p.rewardThisStep = 0;
        if (this.done) return { rewards: this.players.map(() => 0), done: true };

        // 1) Commit pending actions for cell-aligned players.
        this.commitActions();

        // 2) Move (continuous toward target cell).
        this.movePlayers(SIM_DT_MS);

        // 3) Tick bomb fuses; detonate those that just expired (chained synchronously).
        this.advanceBombs(SIM_DT_MS);

        // 4) Advance time BEFORE processing activations, so a brand-new explosion
        //    whose center arrives at elapsedMs gets its arrival event this tick.
        this.elapsedMs += SIM_DT_MS;

        // 5) Process active explosions: wood destruction + kill checks at arrival
        //    and again at end-of-window.
        this.processActiveExplosions();

        // 6) Collect powerups (after possible new ones spawned from destroyed wood).
        this.collectPowerups();

        // 7) Timers.
        this.updateTimers(SIM_DT_MS);

        // 8) Per-step survival penalty so STAY isn't always the safest pick.
        for (const p of this.players) if (p.alive) p.rewardThisStep -= 0.008;

        this.done = this.gameOver();
        return { rewards: this.players.map(p => p.rewardThisStep), done: this.done };
    }
}
