// Thin RL-env wrapper around Sim. Handles:
//  - building per-player ObsView so the same encoder works in training and inference
//  - skipping ticks until the controlled player is at a cell-aligned position
//    (the only point where a new action takes effect)

import { Sim, SimConfig, Action, ACTION_STAY, BOARD_H, BOARD_W, Bomb } from './sim';
import { encode, ObsView } from './observation';

export interface StepResult {
    rewards: number[];
    done: boolean;
    obs: Float32Array[]; // one per player
    cellAligned: boolean[]; // true if player is at a cell where action can be applied
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
                .map(p => ({ y: p.y, x: p.x, alive: p.alive, knocked: p.knocked })),
            woodLeft: this.sim.woodLeft,
        };
    }

    // Apply each player's action (or STAY for dead/knocked/non-aligned) and step once.
    step(actions: Action[]): StepResult {
        for (let i = 0; i < this.sim.players.length; i++) {
            const p = this.sim.players[i];
            if (!p.alive || p.knocked) continue;
            if (this.sim.isAtCell(i)) {
                this.sim.setAction(i, actions[i]);
            } else {
                // Keep the prior pendingAction in motion; ignore new actions mid-cell.
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
