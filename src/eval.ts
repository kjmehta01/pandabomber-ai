// Win-rate eval harness. Plays N deterministic games (ε=0, fixed seeds) between
// a primary model and a list of baselines; reports win rate + behavioral metrics
// per matchup. Seats alternate every other game so corner/first-mover bias cancels.
//
// Baselines:
//   noop                — always STAY (sanity floor)
//   random              — uniform random (seeded per game)
//   model:path/to/json  — another trained checkpoint (regression check)
//
// Usage:
//   npx ts-node src/eval.ts --model=checkpoints/latest.json \
//                           --vs=noop,random,model:checkpoints/prev.json \
//                           --games=100

import './tfBackend'; // must come first — hijacks tfjs-node if TFJS_GPU=1
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import { Env } from './env';
import { Sim, Action, ACTION_STAY, NUM_ACTIONS } from './sim';
import { buildModel, importWeights, obsToTensors } from './model';

const MAX_STEPS_PER_GAME = 15_000; // 150s @ 10ms — margin past the 120s in-game cap

interface Agent {
    name: string;
    selectAction: (obs: Float32Array, sim: Sim, seatIdx: number) => Action;
    dispose?: () => void;
}

function makeNoopAgent(): Agent {
    return { name: 'noop', selectAction: () => ACTION_STAY };
}

function makeRandomAgent(seed: number): Agent {
    // Seeded so re-runs with the same flags are bit-identical.
    let s = seed | 0 || 1;
    const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return ((s >>> 0) / 0x100000000); };
    return { name: 'random', selectAction: () => Math.floor(rnd() * NUM_ACTIONS) as Action };
}

async function makeModelAgent(modelPath: string, label: string): Promise<Agent> {
    if (!fs.existsSync(modelPath)) throw new Error(`model file not found: ${modelPath}`);
    const model = buildModel();
    await importWeights(model, modelPath);
    return {
        name: label,
        selectAction: (obs) => tf.tidy(() => {
            const [spatial, scalar] = obsToTensors(obs);
            const q = model.predict([spatial, scalar]) as tf.Tensor;
            const data = q.dataSync();
            let best = 0, bestV = data[0];
            for (let i = 1; i < NUM_ACTIONS; i++) if (data[i] > bestV) { bestV = data[i]; best = i; }
            return best as Action;
        }),
        dispose: () => model.dispose(),
    };
}

interface PlayerStats {
    bombsPlaced: number;
    woodDestroyed: number;
    powerupsCollected: number;
    killsScored: number;
    diedFromOwnBomb: number;
}

interface GameResult {
    winnerSeat: number;        // -1 if no clear winner
    durationMs: number;
    stats: PlayerStats[];
}

function runGame(agents: Agent[], seed: number): GameResult {
    const env = new Env({
        numPlayers: agents.length,
        seed,
        maxTimeMs: 120_000,
    });
    let obs = env.initialObs();
    let done = false;
    let steps = 0;
    while (!done && steps < MAX_STEPS_PER_GAME) {
        const actions: Action[] = [];
        for (let i = 0; i < agents.length; i++) {
            const p = env.sim.players[i];
            if (env.sim.isAtCell(i) && p.alive && !p.knocked) {
                actions.push(agents[i].selectAction(obs[i], env.sim, i));
            } else {
                actions.push(p.pendingAction);
            }
        }
        const res = env.step(actions);
        obs = res.obs;
        done = res.done;
        steps++;
    }
    // gameOver() always unshifts the winner first, so ranking[0] is the winner.
    const winnerSeat = env.sim.ranking.length > 0 ? env.sim.ranking[0] : -1;
    return {
        winnerSeat,
        durationMs: env.sim.elapsedMs,
        stats: env.sim.players.map(p => ({ ...p.stats })),
    };
}

interface MatchAggregate {
    primary: string;
    opponent: string;
    games: number;
    primaryWins: number;
    opponentWins: number;
    draws: number;
    avgDurationMs: number;
    avgWoodDestroyed: number;
    avgPowerupsCollected: number;
    avgBombsPlaced: number;
    avgKillsScored: number;
    suicideRate: number;
}

function runMatch(primary: Agent, opponent: Agent, games: number, baseSeed: number): MatchAggregate {
    const agg: MatchAggregate = {
        primary: primary.name, opponent: opponent.name, games,
        primaryWins: 0, opponentWins: 0, draws: 0,
        avgDurationMs: 0, avgWoodDestroyed: 0, avgPowerupsCollected: 0,
        avgBombsPlaced: 0, avgKillsScored: 0, suicideRate: 0,
    };
    for (let g = 0; g < games; g++) {
        const primarySeat = g % 2;
        const agents = primarySeat === 0 ? [primary, opponent] : [opponent, primary];
        const result = runGame(agents, baseSeed + g);
        const primaryStats = result.stats[primarySeat];

        if (result.winnerSeat === primarySeat) agg.primaryWins++;
        else if (result.winnerSeat === -1) agg.draws++;
        else agg.opponentWins++;

        agg.avgDurationMs += result.durationMs;
        agg.avgWoodDestroyed += primaryStats.woodDestroyed;
        agg.avgPowerupsCollected += primaryStats.powerupsCollected;
        agg.avgBombsPlaced += primaryStats.bombsPlaced;
        agg.avgKillsScored += primaryStats.killsScored;
        agg.suicideRate += primaryStats.diedFromOwnBomb;
    }
    agg.avgDurationMs /= games;
    agg.avgWoodDestroyed /= games;
    agg.avgPowerupsCollected /= games;
    agg.avgBombsPlaced /= games;
    agg.avgKillsScored /= games;
    agg.suicideRate /= games;
    return agg;
}

interface Args {
    model: string;
    vs: string;        // comma-separated baselines
    games: number;
    seed: number;
    json: string;      // optional output path
}

function parseArgs(): Args {
    const a: Args = {
        model: 'checkpoints/latest.json',
        vs: 'noop,random',
        games: 100,
        seed: 1000,
        json: '',
    };
    for (const arg of process.argv.slice(2)) {
        const m = arg.match(/^--([^=]+)=(.*)$/);
        if (!m) continue;
        const [, k, v] = m;
        if (k in a) {
            const isNum = typeof (a as any)[k] === 'number';
            (a as any)[k] = isNum ? Number(v) : v;
        }
    }
    return a;
}

async function buildOpponent(spec: string, gameOffset: number): Promise<Agent> {
    if (spec === 'noop') return makeNoopAgent();
    if (spec === 'random') return makeRandomAgent(gameOffset);
    if (spec.startsWith('model:')) return makeModelAgent(spec.slice('model:'.length), spec);
    throw new Error(`unknown opponent spec: ${spec} (expected noop|random|model:path)`);
}

function pct(x: number): string {
    return (x * 100).toFixed(1) + '%';
}
function fmt(x: number, w: number = 6): string {
    return x.toFixed(2).padStart(w);
}

function printReport(args: Args, primaryName: string, matches: MatchAggregate[]) {
    console.log('');
    console.log(`=== eval: ${primaryName} (${args.games} games per matchup, base seed ${args.seed}) ===`);
    console.log('');
    const header = ['opponent', 'wins', 'losses', 'draws', 'win_rate', 'avg_dur_s', 'avg_wood', 'avg_pups', 'avg_kills', 'suicide%'];
    console.log(header.map(h => h.padStart(11)).join(' '));
    for (const m of matches) {
        const winRate = m.primaryWins / m.games;
        const row = [
            m.opponent.padStart(11),
            String(m.primaryWins).padStart(11),
            String(m.opponentWins).padStart(11),
            String(m.draws).padStart(11),
            pct(winRate).padStart(11),
            fmt(m.avgDurationMs / 1000).padStart(11),
            fmt(m.avgWoodDestroyed).padStart(11),
            fmt(m.avgPowerupsCollected).padStart(11),
            fmt(m.avgKillsScored).padStart(11),
            pct(m.suicideRate).padStart(11),
        ];
        console.log(row.join(' '));
    }
    console.log('');
}

async function main() {
    const args = parseArgs();
    console.log('[eval] args:', args);

    const primary = await makeModelAgent(args.model, `model[${args.model}]`);
    const opponentSpecs = args.vs.split(',').map(s => s.trim()).filter(Boolean);

    const matches: MatchAggregate[] = [];
    for (const spec of opponentSpecs) {
        // Distinct seed per matchup so the random agent doesn't replay the same
        // move sequence in every matchup.
        const seedOffset = matches.length * args.games * 7;
        const opponent = await buildOpponent(spec, args.seed + seedOffset);
        const t0 = Date.now();
        const agg = runMatch(primary, opponent, args.games, args.seed + seedOffset);
        const t1 = Date.now();
        console.log(`[eval] ${spec}: ${agg.primaryWins}/${args.games} wins (${pct(agg.primaryWins / agg.games)}) in ${((t1 - t0) / 1000).toFixed(1)}s`);
        matches.push(agg);
        opponent.dispose?.();
    }
    primary.dispose?.();

    printReport(args, primary.name, matches);
    if (args.json) {
        fs.writeFileSync(args.json, JSON.stringify({ args, primary: primary.name, matches }, null, 2));
        console.log(`[eval] wrote JSON results → ${args.json}`);
    }
}

main().catch(err => { console.error(err); process.exit(1); });
