// RL wrapper around Sim. Builds per-player ObsView so the same encoder runs in
// training and inference; a new action only takes effect at cell-aligned ticks.

import { Sim, SimConfig, Action, ACTION_STAY, BOARD_H, BOARD_W, Bomb } from './sim';
import { encode, ObsView } from './observation';

export interface StepResult {
    rewards: number[];
    done: boolean;
    obs: Float32Array[];
    cellAligned: boolean[];
}

export class Env {
    sim: Sim;

    constructor(cfg: SimConfig) {
        this.sim = new Sim(cfg);
    }

    private buildView(playerIdx: number): ObsView {
        const me = this.sim.players[playerIdx];
        const bombsFlat: Bomb[] = [];
        for (let r = 0; r < BOARD_H; r++) {
            for (let c = 0; c < BOARD_W; c++) {
                const b = this.sim.bombs[r][c];
                if (b) bombsFlat.push(b);
            }
        }
        const activeBlasts: Array<{ row: number; col: number; msUntilSafe: number }> = [];
        for (const exp of this.sim.activeExplosions) {
            for (const [key, cell] of exp.cells) {
                if (cell.secondFired) continue;
                const [r, c] = key.split(',').map(Number);
                const msUntilSafe = Math.max(0, cell.secondCheckMs - this.sim.elapsedMs);
                activeBlasts.push({ row: r, col: c, msUntilSafe });
            }
        }
        return {
            boardH: BOARD_H,
            boardW: BOARD_W,
            getCell: (r: number, c: number) => this.sim.getCell(r, c),
            bombs: bombsFlat.map(b => ({ row: b.row, col: b.col, power: b.power, fuseRemainingMs: b.fuseRemainingMs })),
            self: {
                y: me.y, x: me.x,
                bombPower: me.bombPower, maxBombs: me.maxBombs,
                moveSpeed: me.moveSpeed, placedBombs: me.placedBombs,
            },
            enemies: this.sim.players
                .filter(p => p.idx !== playerIdx)
                .map(p => ({ y: p.y, x: p.x, alive: p.alive, knocked: p.knocked, knockMsLeft: p.knockMsLeft })),
            activeBlasts,
            woodLeft: this.sim.woodLeft,
        };
    }

    step(actions: Action[]): StepResult {
        for (let i = 0; i < this.sim.players.length; i++) {
            const p = this.sim.players[i];
            if (!p.alive || p.knocked) continue;
            // Mid-cell ticks re-commit pendingAction; new actions only apply at cell alignment.
            if (this.sim.isAtCell(i)) {
                this.sim.setAction(i, actions[i]);
            } else {
                this.sim.setAction(i, p.pendingAction);
            }
        }
        const { rewards, done } = this.sim.step();
        const obs: Float32Array[] = [];
        const cellAligned: boolean[] = [];
        for (let i = 0; i < this.sim.players.length; i++) {
            obs.push(encode(this.buildView(i)));
            cellAligned.push(this.sim.isAtCell(i));
        }
        return { rewards, done, obs, cellAligned };
    }

    initialObs(): Float32Array[] {
        return this.sim.players.map((_, i) => encode(this.buildView(i)));
    }

    numPlayers(): number {
        return this.sim.players.length;
    }
}
