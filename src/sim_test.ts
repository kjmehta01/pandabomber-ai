// Smoke tests for the sim's explosion-timing model.
//   npx ts-node src/sim_test.ts
//
// Covers: per-cell arrival times (+i*25ms), far-end dodge window, chain detonation
// sharing one ActiveExplosion, and cell-safe-after-window pruning.

import { Sim, SIM_DT_MS, ACTION_STAY, ACTION_RIGHT, ACTION_BOMB, EXPLOSION_TRAVEL_MS, DEATH_CHECK_WINDOW_MS, BOMB_FUSE_MS } from './sim';

function assert(cond: boolean, msg: string) {
    if (!cond) { console.error('FAIL:', msg); process.exit(1); }
    else console.log('  ok:', msg);
}

function test_explosion_arrival_times() {
    console.log('\n[test] explosion arrival times');
    // 2 players so the game doesn't end mid-test; park the 2nd far from the blast.
    const sim = new Sim({ numPlayers: 2, seed: 42, woodOdds: 0, powerupOdds: 0, maxTimeMs: 60_000 });
    sim.players[1].y = 11; sim.players[1].x = 17;
    for (let r = 1; r < 12; r++) for (let c = 1; c < 18; c++) {
        if (r % 2 === 0 && c % 2 === 0) continue; // keep stone pattern
        (sim.blocks[r] as any)[c] = undefined;
    }
    sim.players[0].y = 1; sim.players[0].x = 1;
    sim.players[0].bombPower = 5;
    sim.players[0].maxBombs = 1;
    sim.setAction(0, ACTION_BOMB);
    sim.step();
    let detonationStep = -1;
    for (let i = 0; i < 500; i++) {
        sim.step();
        if (sim.activeExplosions.length > 0) { detonationStep = i; break; }
    }
    assert(detonationStep >= 0, `explosion appeared within 500 steps`);
    assert(sim.activeExplosions.length === 1, 'one active explosion at fuse expiry');
    const exp = sim.activeExplosions[0];
    const center = exp.cells.get('1,1')!;
    const d1 = exp.cells.get('1,2')!;     // +25ms
    const d4 = exp.cells.get('1,5')!;     // +100ms
    const d5 = exp.cells.get('1,6')!;     // +125ms (max for power=5)
    assert(center !== undefined && d1 !== undefined && d4 !== undefined && d5 !== undefined, 'center/d1/d4/d5 all in explosion');
    assert(d1.arrivalMs - center.arrivalMs === EXPLOSION_TRAVEL_MS, `d1 arrives 25ms after center (got ${d1.arrivalMs - center.arrivalMs})`);
    assert(d4.arrivalMs - center.arrivalMs === 4 * EXPLOSION_TRAVEL_MS, `d4 arrives 100ms after center (got ${d4.arrivalMs - center.arrivalMs})`);
    assert(d5.arrivalMs - center.arrivalMs === 5 * EXPLOSION_TRAVEL_MS, `d5 arrives 125ms after center (got ${d5.arrivalMs - center.arrivalMs})`);
}

function test_far_cell_dodge_window() {
    console.log('\n[test] far-end cell allows more dodge time than near-end');
    // power=10 → cells 1..10 hit at +25ms, +50ms, … +250ms.
    const sim = new Sim({ numPlayers: 2, seed: 1, woodOdds: 0, powerupOdds: 0 });
    sim.players[1].y = 11; sim.players[1].x = 17;
    for (let r = 1; r < 12; r++) for (let c = 1; c < 18; c++) {
        if (r % 2 === 0 && c % 2 === 0) continue;
        (sim.blocks[r] as any)[c] = undefined;
    }
    sim.players[0].y = 1; sim.players[0].x = 1;
    sim.players[0].bombPower = 10;
    sim.setAction(0, ACTION_BOMB);
    sim.step();
    for (let i = 0; i < 500; i++) { sim.step(); if (sim.activeExplosions.length > 0) break; }
    const exp = sim.activeExplosions[0];
    const d1 = exp.cells.get('1,2');
    const d10 = exp.cells.get('1,11');
    assert(d1 !== undefined && d10 !== undefined, 'both d1 and d10 in the blast');
    assert(d10!.arrivalMs - d1!.arrivalMs === 9 * EXPLOSION_TRAVEL_MS, `d10 arrives 225ms after d1 (got ${d10!.arrivalMs - d1!.arrivalMs})`);
}

function test_chain_detonation_simultaneous() {
    console.log('\n[test] chained bombs both detonate at fuse expiry of the first');
    const sim = new Sim({ numPlayers: 2, seed: 7, woodOdds: 0, powerupOdds: 0 });
    for (let r = 1; r < 12; r++) for (let c = 1; c < 18; c++) {
        if (r % 2 === 0 && c % 2 === 0) continue;
        (sim.blocks[r] as any)[c] = undefined;
    }
    sim.players[0].y = 1; sim.players[0].x = 1; sim.players[0].bombPower = 4;
    sim.players[1].y = 1; sim.players[1].x = 5; sim.players[1].bombPower = 2;
    // P0's bomb at (1,1) is placed first; its ray chains P1's bomb at (1,5).
    sim.setAction(0, ACTION_BOMB); sim.step();
    sim.setAction(1, ACTION_BOMB); sim.step();
    for (let i = 0; i < 500; i++) { sim.step(); if (sim.activeExplosions.length > 0) break; }
    assert(sim.activeExplosions.length === 1, 'chained bombs share one ActiveExplosion');
    const exp = sim.activeExplosions[0];
    const c0 = exp.cells.get('1,1');
    const c1 = exp.cells.get('1,5');
    assert(c0 !== undefined && c1 !== undefined, 'both centers present');
    assert(c0!.arrivalMs === c1!.arrivalMs, `chained centers share arrival time (got ${c0!.arrivalMs} vs ${c1!.arrivalMs})`);
    const d6 = exp.cells.get('1,6');
    const d7 = exp.cells.get('1,7');
    assert(d6 !== undefined && d7 !== undefined, 'chained bomb extends ray past its center');
    assert(d6!.arrivalMs === c1!.arrivalMs + 25, '(1,6) is +25ms from chained center');
}

function test_player_at_far_end_can_dodge() {
    console.log('\n[test] player at distance 10 has time to step away from ray arrival');
    const sim = new Sim({ numPlayers: 2, seed: 3, woodOdds: 0, powerupOdds: 0 });
    sim.players[1].y = 11; sim.players[1].x = 17;
    for (let r = 1; r < 12; r++) for (let c = 1; c < 18; c++) {
        if (r % 2 === 0 && c % 2 === 0) continue;
        (sim.blocks[r] as any)[c] = undefined;
    }
    sim.players[0].y = 1; sim.players[0].x = 11; // far end of a power=10 bomb at (1,1)
    sim.players[0].bombPower = 1;
    // Inject the bomb directly; ownership doesn't change kill semantics here.
    sim.bombs[1][1] = { row: 1, col: 1, power: 10, fuseRemainingMs: SIM_DT_MS, ownerIdx: 0 };
    sim.players[0].placedBombs = 1;
    sim.step();
    let killed = false;
    // Distance-10 ray arrives ≥26 ticks after detonation (250ms / 10ms).
    for (let i = 0; i < 60; i++) {
        sim.step();
        if (!sim.players[0].alive) { killed = true; break; }
        if (sim.players[0].knocked) { killed = true; break; }
    }
    assert(killed, 'standing player at distance 10 eventually gets hit when ray arrives');
}

function test_cell_safe_after_kill_window() {
    console.log('\n[test] cell is safe to walk onto once its kill window has closed');
    const sim = new Sim({ numPlayers: 2, seed: 11, woodOdds: 0, powerupOdds: 0 });
    sim.players[1].y = 11; sim.players[1].x = 17;
    for (let r = 1; r < 12; r++) for (let c = 1; c < 18; c++) {
        if (r % 2 === 0 && c % 2 === 0) continue;
        (sim.blocks[r] as any)[c] = undefined;
    }
    // Park player 0 well outside any blast ray.
    sim.players[0].y = 5; sim.players[0].x = 17; sim.players[0].bombPower = 5;
    sim.bombs[1][1] = { row: 1, col: 1, power: 5, fuseRemainingMs: SIM_DT_MS, ownerIdx: 0 };
    sim.players[0].placedBombs = 1;
    sim.step();
    assert(sim.activeExplosions.length === 1, 'explosion live right after detonation');

    // Distance-5 cell's window ends at detTime+175ms ⇒ ≥18 ticks; give margin.
    for (let i = 0; i < 40; i++) sim.step();
    assert(sim.activeExplosions.length === 0, `explosion pruned after window closes (still ${sim.activeExplosions.length})`);

    sim.players[0].y = 1; sim.players[0].x = 2;
    for (let i = 0; i < 30; i++) sim.step();
    assert(sim.players[0].alive && !sim.players[0].knocked, 'player walks onto stale blast cell unharmed');
}

test_explosion_arrival_times();
test_far_cell_dodge_window();
test_chain_detonation_simultaneous();
test_player_at_far_end_can_dodge();
test_cell_safe_after_kill_window();
console.log('\nall tests passed');
