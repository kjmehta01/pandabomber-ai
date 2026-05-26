// Pure-logic MU simulator that tracks production game.ts / gameplayer.ts at 10ms
// ticks (production's 100Hz tickRate). TT (time-trial) is out of scope — the bot is
// only ever deployed in 4-player MU.
//
// Exactly modeled vs production:
//   - bomb fuse (3000ms), invulnerability (200ms), knock duration (6000ms)
//   - 3-strikes-same-cell ⇒ outright death (gameplayer.ts:70-84)
//   - chain detonation: a ray hitting a bomb chains synchronously and PASSES
//     THROUGH the bomb's cell (production explodeBomb has no break post-chain)
//   - rays stop at stone/wood; wood is destroyed AT THE RAY'S ARRIVAL TIME, not
//     at fuse-expiry (production recurseExecute)
//   - 25ms/cell explosion travel; per-cell death window = [arrival, arrival+50ms]
//     (production fires checkPlayerDeaths twice per cell)
//   - powerup caps (13), MU spawn odds, death-drop spawn (4 in 5×5 around death)
//   - corner spawn clearing (3 cells per corner are wood-free)
//
// Board size is configurable per Sim (SimConfig.boardH/boardW). The observation
// encoder stays fixed at 13×19 — egocentric padding handles smaller boards, so
// the trained model ports to production unchanged.
export const DEFAULT_BOARD_H = 13;
export const DEFAULT_BOARD_W = 19;

export const SIM_DT_MS = 10;
export const BOMB_FUSE_MS = 3000;
export const KNOCK_DURATION_MS = 6000;
export const INVULNERABILITY_MS = 200;
export const MAX_POWERUPS = 13;
export const EXPLOSION_TRAVEL_MS = 25; // game.ts:12
export const DEATH_CHECK_WINDOW_MS = 50; // 2nd checkPlayerDeaths is +50ms in recurseExecute

// Aggression-tuned: kills/engagement dominate; wood is a means, not a goal.
// No illegal-action penalty — rollout masking (legalActionMask) handles it.
const REWARD_WOOD = 0.2;
const REWARD_POWERUP = 0.4;
const REWARD_KNOCK_SCORED = 4.0;
const REWARD_KNOCK_RECEIVED = -0.3;
const REWARD_KNOCK_SELF = -1.0;
const REWARD_KILL_SCORED = 20.0;
const REWARD_DEATH = -2.0;
const REWARD_DEATH_SELF = -4.0;
const REWARD_LAST_ALIVE = 10.0;
const REWARD_TIMEOUT_SURVIVOR = -3.0;
const REWARD_PER_TICK_ALIVE = -0.0005;
// Paid at bomb placement when an enemy is inside the bomb's cardinal ray
// (walls block). Pushes the policy to bomb AT opponents, not just at wood.
const REWARD_BOMB_NEAR_ENEMY = 0.6;
// Signed shaping per cell of Manhattan-distance change to nearest enemy.
const REWARD_APPROACH_PER_CELL = 0.02;

const START_MOVE_SPEED = 0.045;
const MOVE_SPEED_INCREMENT = 0.003;
// Production: cells/frame = speed*1.667 at 60Hz ⇒ cells/ms = speed*1.667*60/1000.
const CELLS_PER_MS = (speed: number) => speed * 1.667 * 60 / 1000;

// Delay after woodLeft first hits 0 (or sim start, if no wood) before random
// ownerless bombs/powerups begin spawning.
const DEFAULT_RANDOM_SPAWN_DELAY_MS = 30_000;
const RANDOM_SPAWN_INTERVAL_MS = 2_000;
const RANDOM_BOMB_POWER = 3;

export type Cell = 'E' | 'S' | 'W' | 'B' | 'NUM' | 'SPE' | 'STR';
export type Action = 0 | 1 | 2 | 3 | 4 | 5;
export const ACTION_STAY = 0;
export const ACTION_UP = 1;
export const ACTION_DOWN = 2;
export const ACTION_LEFT = 3;
export const ACTION_RIGHT = 4;
export const ACTION_BOMB = 5;
export const NUM_ACTIONS = 6;

// Sentinel ownerIdx for ownerless (random/environmental) bombs.
export const OWNER_RANDOM = -1;

export interface Bomb {
    row: number;
    col: number;
    power: number;
    fuseRemainingMs: number;
    ownerIdx: number;  // OWNER_RANDOM (-1) for random spawns
}

export interface SimPlayer {
    idx: number;
    alive: boolean;
    knocked: boolean;
    immuneMs: number;
    knockMsLeft: number;
    y: number;
    x: number;
    // Sign of motion (-1, 0, 1); magnitude from moveSpeed via CELLS_PER_MS.
    dyDir: number;
    dxDir: number;
    // Target cell while in transit; set at commit, cleared on arrival.
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
    // Previous min-Manhattan distance to any alive enemy; drives approach shaping.
    lastEnemyDist: number;
    // Per-game counters for the eval harness; O(1) per event.
    stats: {
        bombsPlaced: number;
        woodDestroyed: number;
        powerupsCollected: number;
        knocksScored: number;
        killsScored: number;
        diedFromOwnBomb: number;
        illegalMoves: number;
        illegalBombs: number;
        knocksReceived: number;
        // Episode-cumulative reward broken down by source. Sums to totalReward.
        // Useful to diagnose which incentives are dominating the policy.
        rewardBreakdown: {
            wood: number;
            powerup: number;
            knockScored: number;
            knockReceived: number;
            knockSelf: number;
            killScored: number;
            death: number;
            deathSelf: number;
            lastAlive: number;
            timeoutSurvivor: number;
            perTick: number;
            bombNearEnemy: number;
            approach: number;
        };
    };
}

// One active explosion. `cells` keys are "r,c". Production fires two kill checks
// per cell (at arrival and arrival+50ms), then the cell is permanently safe.
// On collisions (chains hitting the same cell), the EARLIEST arrival wins —
// invulnerability + 50ms windows make later rays redundant anyway.
export interface ExplosionCell {
    arrivalMs: number;
    secondCheckMs: number;
    sourceOwnerIdx: number; // OWNER_RANDOM (-1) for ownerless bombs
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
    boardH?: number;
    boardW?: number;
    woodOdds?: number;
    powerupOdds?: number;
    seed?: number;
    maxTimeMs?: number;
    // Starting powerup tier per category (1 = vanilla, capped at MAX_POWERUPS).
    startBombPower?: number;
    startMaxBombs?: number;
    startSpeedTier?: number;
    // Spawn ownerless bombs/powerups `randomSpawnDelayMs` after woodLeft first
    // hits 0 (or after sim start when woodOdds=0).
    randomSpawnsEnabled?: boolean;
    randomSpawnDelayMs?: number;
    mapName?: string;
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
    cfg: Required<Omit<SimConfig, 'mapName'>> & { mapName: string };
    boardH: number;
    boardW: number;
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
    // Set the first tick woodLeft hits 0 (or at construction if it starts at 0).
    woodGoneMs: number | null;
    lastRandomSpawnMs: number;

    constructor(cfg: SimConfig) {
        const startBombPower = cfg.startBombPower ?? 1;
        const startMaxBombs = cfg.startMaxBombs ?? 1;
        const startSpeedTier = cfg.startSpeedTier ?? 1;
        this.cfg = {
            numPlayers: cfg.numPlayers,
            boardH: cfg.boardH ?? DEFAULT_BOARD_H,
            boardW: cfg.boardW ?? DEFAULT_BOARD_W,
            woodOdds: cfg.woodOdds ?? 0.8,
            powerupOdds: cfg.powerupOdds ?? 0.4,
            seed: cfg.seed ?? Math.floor(Math.random() * 1e9),
            maxTimeMs: cfg.maxTimeMs ?? 180_000,
            startBombPower,
            startMaxBombs,
            startSpeedTier,
            randomSpawnsEnabled: cfg.randomSpawnsEnabled ?? false,
            randomSpawnDelayMs: cfg.randomSpawnDelayMs ?? DEFAULT_RANDOM_SPAWN_DELAY_MS,
            mapName: cfg.mapName ?? `${cfg.boardH ?? DEFAULT_BOARD_H}x${cfg.boardW ?? DEFAULT_BOARD_W}`,
        };
        this.boardH = this.cfg.boardH;
        this.boardW = this.cfg.boardW;
        this.rng = makeRng(this.cfg.seed);

        this.blocks = [];
        this.bombs = [];
        this.powerups = [];
        this.activeExplosions = [];
        for (let r = 0; r < this.boardH; r++) {
            this.blocks.push(new Array(this.boardW).fill(undefined));
            this.bombs.push(new Array(this.boardW).fill(undefined));
            this.powerups.push(new Array(this.boardW).fill(undefined));
        }

        for (let r = 0; r < this.boardH; r++) {
            for (let c = 0; c < this.boardW; c++) {
                if (r === 0 || r === this.boardH - 1 || c === 0 || c === this.boardW - 1) {
                    this.blocks[r][c] = 'S';
                } else if (r % 2 === 0 && c % 2 === 0) {
                    this.blocks[r][c] = 'S';
                }
            }
        }

        const corners: [number, number][][] = [
            [[1, 1], [1, 2], [2, 1]],
            [[1, this.boardW - 2], [1, this.boardW - 3], [2, this.boardW - 2]],
            [[this.boardH - 2, 1], [this.boardH - 2, 2], [this.boardH - 3, 1]],
            [[this.boardH - 2, this.boardW - 2], [this.boardH - 2, this.boardW - 3], [this.boardH - 3, this.boardW - 2]],
        ];
        const cornerSet = new Set<string>();
        for (const c of corners) for (const [r, x] of c) cornerSet.add(`${r},${x}`);

        this.woodLeft = 0;
        for (let r = 1; r < this.boardH - 1; r++) {
            for (let c = 1; c < this.boardW - 1; c++) {
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
            [1, this.boardW - 2],
            [this.boardH - 2, 1],
            [this.boardH - 2, this.boardW - 2],
        ];
        const order = [0, 1, 2, 3];
        for (let i = order.length - 1; i > 0; i--) {
            const j = Math.floor(this.rng() * (i + 1));
            [order[i], order[j]] = [order[j], order[i]];
        }

        // Tier N = N-1 free pickups of that type.
        const startMoveSpeed = START_MOVE_SPEED + Math.max(0, startSpeedTier - 1) * MOVE_SPEED_INCREMENT;
        const startBombPowerClamped = Math.min(MAX_POWERUPS, Math.max(1, startBombPower));
        const startMaxBombsClamped = Math.min(MAX_POWERUPS, Math.max(1, startMaxBombs));

        this.players = [];
        for (let i = 0; i < this.cfg.numPlayers; i++) {
            const [py, px] = corner[order[i]];
            this.players.push({
                idx: i, alive: true, knocked: false, immuneMs: 0, knockMsLeft: 0,
                y: py, x: px, dyDir: 0, dxDir: 0, moveTargetY: py, moveTargetX: px,
                pendingAction: ACTION_STAY,
                moveSpeed: startMoveSpeed,
                bombPower: startBombPowerClamped,
                maxBombs: startMaxBombsClamped,
                placedBombs: 0,
                lastDeathRow: -1, lastDeathCol: -1, numDies: 0, firstDieTimeMs: -1,
                rewardThisStep: 0,
                lastEnemyDist: -1,
                stats: {
                    bombsPlaced: 0, woodDestroyed: 0, powerupsCollected: 0, knocksScored: 0, killsScored: 0,
                    diedFromOwnBomb: 0, illegalMoves: 0, illegalBombs: 0, knocksReceived: 0,
                    rewardBreakdown: {
                        wood: 0, powerup: 0, knockScored: 0, knockReceived: 0, knockSelf: 0, killScored: 0,
                        death: 0, deathSelf: 0, lastAlive: 0, timeoutSurvivor: 0, perTick: 0,
                        bombNearEnemy: 0, approach: 0,
                    },
                },
            });
        }
        this.elapsedMs = 0;
        this.ranking = [];
        this.done = false;
        this.woodGoneMs = this.woodLeft === 0 ? 0 : null;
        // Negative so the first spawn fires the tick the delay window opens.
        this.lastRandomSpawnMs = -RANDOM_SPAWN_INTERVAL_MS;
    }

    private addReward(p: SimPlayer, source: keyof SimPlayer['stats']['rewardBreakdown'], amount: number) {
        p.rewardThisStep += amount;
        p.stats.rewardBreakdown[source] += amount;
    }

    getCell(r: number, c: number): Cell {
        if (r < 0 || r >= this.boardH || c < 0 || c >= this.boardW) return 'S';
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

    // True iff the player is at integer cell coords — the only point where a new
    // action can take effect.
    isAtCell(playerIdx: number): boolean {
        const p = this.players[playerIdx];
        return p.dyDir === 0 && p.dxDir === 0;
    }

    // Mask of which actions the sim would accept right now for this player.
    // 1 = legal, 0 = would be rejected (with the illegal-action penalty). Used at
    // rollout time so the policy never picks structurally-impossible actions —
    // breaks the STAY-collapse equilibrium where illegal penalties dominate the
    // Q-landscape and the agent's safest pick is to do nothing.
    legalActionMask(playerIdx: number): Uint8Array {
        const mask = new Uint8Array(NUM_ACTIONS);
        mask[ACTION_STAY] = 1;
        const p = this.players[playerIdx];
        if (!p.alive || p.knocked) return mask;
        const r = Math.round(p.y);
        const c = Math.round(p.x);
        for (let a = 1; a <= 4; a++) {
            const tr = r + DIR_DY[a];
            const tc = c + DIR_DX[a];
            const cell = this.getCell(tr, tc);
            if (cell !== 'S' && cell !== 'W' && cell !== 'B') mask[a] = 1;
        }
        if (p.placedBombs < p.maxBombs && !this.bombs[r][c] && !this.blocks[r][c]) {
            mask[ACTION_BOMB] = 1;
        }
        return mask;
    }

    setAction(playerIdx: number, action: Action) {
        const p = this.players[playerIdx];
        if (!p.alive) return;
        p.pendingAction = action;
    }

    private commitActions() {
        for (const p of this.players) {
            if (!p.alive || p.knocked) continue;
            if (p.dyDir !== 0 || p.dxDir !== 0) continue; // mid-cell
            const a = p.pendingAction;
            p.pendingAction = ACTION_STAY;
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
            const target = this.getCell(tr, tc);
            // Reject walks into walls/wood/bombs/oob. With action masking these
            // should be unreachable from rollout; the counter stays as a diagnostic
            // tripwire if masking ever drifts.
            if (target === 'S' || target === 'W' || target === 'B') {
                p.stats.illegalMoves++;
                continue;
            }
            p.moveTargetY = tr;
            p.moveTargetX = tc;
            p.dyDir = dy;
            p.dxDir = dx;
        }
    }

    // True if (br,bc) with `power` has any non-self alive enemy in its cardinal
    // ray (stone/wood block). Drives REWARD_BOMB_NEAR_ENEMY.
    private bombCoversEnemy(br: number, bc: number, power: number, ownerIdx: number): boolean {
        for (const [dy, dx] of DIRS) {
            for (let i = 1; i <= power; i++) {
                const r = br + dy * i;
                const c = bc + dx * i;
                if (r < 0 || r >= this.boardH || c < 0 || c >= this.boardW) break;
                if (this.blocks[r][c] === 'S') break;
                if (this.blocks[r][c] === 'W') break;
                for (const e of this.players) {
                    if (!e.alive || e.idx === ownerIdx) continue;
                    if (Math.round(e.y) === r && Math.round(e.x) === c) return true;
                }
            }
        }
        return false;
    }

    private placeBomb(p: SimPlayer, r: number, c: number) {
        if (p.placedBombs >= p.maxBombs || this.bombs[r][c] || this.blocks[r][c]) {
            p.stats.illegalBombs++;
            return;
        }
        this.bombs[r][c] = { row: r, col: c, power: p.bombPower, fuseRemainingMs: BOMB_FUSE_MS, ownerIdx: p.idx };
        p.placedBombs++;
        p.stats.bombsPlaced++;
        // Paid at placement (not detonation) so the bonus survives a dodge.
        if (this.bombCoversEnemy(r, c, p.bombPower, p.idx)) {
            this.addReward(p, 'bombNearEnemy', REWARD_BOMB_NEAR_ENEMY);
        }
    }

    // Ownerless: doesn't touch any player's bomb cap; damage/wood credit skipped.
    private placeRandomBomb(r: number, c: number) {
        if (this.bombs[r][c] || this.blocks[r][c]) return;
        this.bombs[r][c] = { row: r, col: c, power: RANDOM_BOMB_POWER, fuseRemainingMs: BOMB_FUSE_MS, ownerIdx: OWNER_RANDOM };
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
        for (let r = 0; r < this.boardH; r++) {
            for (let c = 0; c < this.boardW; c++) {
                const b = this.bombs[r][c];
                if (b) b.fuseRemainingMs -= dtMs;
            }
        }
        const processed = new Set<Bomb>();
        for (let r = 0; r < this.boardH; r++) {
            for (let c = 0; c < this.boardW; c++) {
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

    // Mirrors production explodeBomb's recursive chain: a ray hitting a bomb
    // chains it (no break), but wood/stone stop the ray and so also stop the chain.
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
                    if (r < 0 || r >= this.boardH || c < 0 || c >= this.boardW) break;
                    if (this.blocks[r][c] === 'S') break;
                    if (this.blocks[r][c] === 'W') break;
                    const other = this.bombs[r][c];
                    if (other && !chain.has(other)) {
                        chain.add(other);
                        queue.push(other);
                    }
                }
            }
        }
        return [...chain];
    }

    // Compute per-cell arrival times from all chain centers; wood destruction and
    // kill checks happen later in processActiveExplosions at each cell's arrival tick.
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
                // Earlier ray wins wood-destroy credit + bounds the first kill check.
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
                    if (r < 0 || r >= this.boardH || c < 0 || c >= this.boardW) break;
                    if (this.blocks[r][c] === 'S') break;
                    const arrivalT = explodeAt + i * EXPLOSION_TRAVEL_MS;
                    upsert(r, c, arrivalT, b.ownerIdx);
                    if (this.blocks[r][c] === 'W') break;
                }
            }
        }

        for (const b of centers) {
            this.bombs[b.row][b.col] = undefined;
            // Ownerless bombs don't count against any player's bomb cap.
            if (b.ownerIdx !== OWNER_RANDOM) {
                this.players[b.ownerIdx].placedBombs--;
            }
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

                // First check at arrival: destroy wood + kill check.
                if (!cell.firstFired && now >= cell.arrivalMs) {
                    cell.firstFired = true;
                    if (this.blocks[r][c] === 'W') {
                        this.blocks[r][c] = undefined;
                        this.woodLeft--;
                        if (cell.sourceOwnerIdx !== OWNER_RANDOM) {
                            this.addReward(this.players[cell.sourceOwnerIdx], 'wood', REWARD_WOOD);
                            this.players[cell.sourceOwnerIdx].stats.woodDestroyed++;
                        }
                        this.maybeSpawnPowerup(r, c);
                        if (this.woodLeft === 0 && this.woodGoneMs === null) {
                            this.woodGoneMs = this.elapsedMs;
                        }
                    }
                    this.killCheck(r, c, cell.sourceOwnerIdx);
                }

                // Second kill check at arrival+50ms.
                if (cell.firstFired && !cell.secondFired && now >= cell.secondCheckMs) {
                    cell.secondFired = true;
                    this.killCheck(r, c, cell.sourceOwnerIdx);
                }

                if (!cell.firstFired || !cell.secondFired) anyUnfired = true;
            }
            // Production stops checking a cell forever after the +50ms timer fires.
            if (anyUnfired) stillActive.push(exp);
        }
        this.activeExplosions = stillActive;
    }

    private killCheck(r: number, c: number, attackerIdx: number) {
        for (const p of this.players) {
            if (!p.alive) continue;
            if (Math.round(p.y) === r && Math.round(p.x) === c) {
                const attacker = attackerIdx === OWNER_RANDOM ? null : this.players[attackerIdx];
                this.damagePlayer(p, attacker);
            }
        }
    }

    /* ─── player damage / death ──────────────────────────────────────── */

    // attacker = null for environmental (random-bomb) damage.
    private damagePlayer(p: SimPlayer, attacker: SimPlayer | null) {
        if (p.immuneMs > 0) return;
        p.immuneMs = INVULNERABILITY_MS;

        const row = Math.round(p.y);
        const col = Math.round(p.x);
        if (!p.knocked) {
            p.knocked = true;
            p.knockMsLeft = KNOCK_DURATION_MS;
            p.dyDir = 0; p.dxDir = 0; // production freezes knocked players
            p.lastEnemyDist = -1;     // avoid stale delta on wake-up

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
            if (attacker !== null && attacker !== p) {
                this.addReward(attacker, 'knockScored', REWARD_KNOCK_SCORED);
                attacker.stats.knocksScored++;
                this.addReward(p, 'knockReceived', REWARD_KNOCK_RECEIVED);
            } else if (attacker === p) {
                this.addReward(p, 'knockSelf', REWARD_KNOCK_SELF);
            } else {
                this.addReward(p, 'knockReceived', REWARD_KNOCK_RECEIVED);
            }
            p.stats.knocksReceived++;
        } else {
            this.kill(p, attacker);
        }
    }

    private kill(p: SimPlayer, attacker: SimPlayer | null) {
        p.alive = false;
        p.knocked = false;
        p.dyDir = 0; p.dxDir = 0;
        if (!this.ranking.includes(p.idx)) this.ranking.unshift(p.idx);
        if (attacker !== null && attacker !== p) {
            this.addReward(attacker, 'killScored', REWARD_KILL_SCORED);
            attacker.stats.killsScored++;
            this.addReward(p, 'death', REWARD_DEATH);
        } else if (attacker === p) {
            p.stats.diedFromOwnBomb = 1;
            this.addReward(p, 'deathSelf', REWARD_DEATH_SELF);
        } else {
            this.addReward(p, 'death', REWARD_DEATH);
        }

        const numDrop = 4;
        const spots: [number, number][] = [];
        for (let dy = -2; dy <= 2; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
                const r = Math.round(p.y) + dy;
                const c = Math.round(p.x) + dx;
                if (r > 0 && r < this.boardH - 1 && c > 0 && c < this.boardW - 1 && this.getCell(r, c) === 'E') {
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
            this.addReward(p, 'powerup', REWARD_POWERUP);
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

    // Interior cells with no block, bomb, powerup, or alive player.
    private listEmptyCellsForRandomSpawn(): [number, number][] {
        const occupied = new Set<number>();
        for (const pl of this.players) {
            if (!pl.alive) continue;
            const r = Math.round(pl.y);
            const c = Math.round(pl.x);
            occupied.add(r * this.boardW + c);
        }
        const out: [number, number][] = [];
        for (let r = 1; r < this.boardH - 1; r++) {
            for (let c = 1; c < this.boardW - 1; c++) {
                if (this.blocks[r][c]) continue;
                if (this.bombs[r][c]) continue;
                if (this.powerups[r][c]) continue;
                if (occupied.has(r * this.boardW + c)) continue;
                out.push([r, c]);
            }
        }
        return out;
    }

    // One ownerless bomb + one powerup at random empty cells, every
    // RANDOM_SPAWN_INTERVAL_MS once the post-wood delay has elapsed.
    private maybeRandomSpawn() {
        if (!this.cfg.randomSpawnsEnabled) return;
        if (this.woodGoneMs === null) return;
        if (this.elapsedMs - this.woodGoneMs < this.cfg.randomSpawnDelayMs) return;
        if (this.elapsedMs - this.lastRandomSpawnMs < RANDOM_SPAWN_INTERVAL_MS) return;
        this.lastRandomSpawnMs = this.elapsedMs;

        const cells = this.listEmptyCellsForRandomSpawn();
        if (cells.length === 0) return;
        const [br, bc] = cells[Math.floor(this.rng() * cells.length)];
        this.placeRandomBomb(br, bc);
        // Re-list — the bomb just consumed one cell.
        const cells2 = this.listEmptyCellsForRandomSpawn();
        if (cells2.length > 0) {
            const [pr, pc] = cells2[Math.floor(this.rng() * cells2.length)];
            this.placePowerupRandom(pr, pc);
        }
    }

    // -1 if no alive enemies.
    private nearestEnemyDist(p: SimPlayer): number {
        let best = -1;
        const pr = Math.round(p.y), pc = Math.round(p.x);
        for (const e of this.players) {
            if (!e.alive || e.idx === p.idx) continue;
            const d = Math.abs(Math.round(e.y) - pr) + Math.abs(Math.round(e.x) - pc);
            if (best < 0 || d < best) best = d;
        }
        return best;
    }

    // Reward = REWARD_APPROACH_PER_CELL × (lastDist − curDist), sampled at
    // cell-aligned ticks. Skip while knocked; damagePlayer clears lastEnemyDist
    // on knock entry so the 6s freeze can't produce a stale delta on wake-up.
    private applyApproachShaping() {
        for (const p of this.players) {
            if (!p.alive || p.knocked) continue;
            if (!this.isAtCell(p.idx)) continue;
            const d = this.nearestEnemyDist(p);
            if (d < 0) { p.lastEnemyDist = -1; continue; }
            if (p.lastEnemyDist >= 0 && d !== p.lastEnemyDist) {
                const delta = p.lastEnemyDist - d; // + = closing, - = retreating
                this.addReward(p, 'approach', REWARD_APPROACH_PER_CELL * delta);
            }
            p.lastEnemyDist = d;
        }
    }

    private gameOver(): boolean {
        const alive = this.players.filter(p => p.alive);
        if (alive.length === 0) return true;
        if (alive.length === 1 && this.cfg.numPlayers > 1) {
            if (!this.ranking.includes(alive[0].idx)) this.ranking.unshift(alive[0].idx);
            // Camping deterrent: only reward last-alive if the survivor actually
            // participated (≥1 knock or kill). Pure survivors get nothing.
            if (alive[0].stats.killsScored + alive[0].stats.knocksScored >= 1) {
                this.addReward(alive[0], 'lastAlive', REWARD_LAST_ALIVE);
            }
            return true;
        }
        if (this.elapsedMs >= this.cfg.maxTimeMs) {
            // Penalize timeout survivors — without this, mutual avoidance becomes
            // viable and the per-step penalty alone doesn't break the stalemate.
            for (const p of this.players) {
                if (p.alive) {
                    this.addReward(p, 'timeoutSurvivor', REWARD_TIMEOUT_SURVIVOR);
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

        this.commitActions();
        this.movePlayers(SIM_DT_MS);
        this.advanceBombs(SIM_DT_MS);
        // Advance time BEFORE processing explosions so a same-tick brand-new
        // explosion's center fires its arrival event this tick.
        this.elapsedMs += SIM_DT_MS;
        this.processActiveExplosions();
        // Collect after explosions so powerups spawned from this tick's wood are pickable.
        this.collectPowerups();
        this.maybeRandomSpawn();
        this.applyApproachShaping();
        this.updateTimers(SIM_DT_MS);

        // Per-step survival penalty so STAY isn't the universally safest pick.
        for (const p of this.players) if (p.alive) this.addReward(p, 'perTick', REWARD_PER_TICK_ALIVE);

        this.done = this.gameOver();
        return { rewards: this.players.map(p => p.rewardThisStep), done: this.done };
    }
}
