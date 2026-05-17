// Self-play Double-DQN training loop with an opponent pool.
//
// Outline:
//  - Three CNN models: `online` (learner, ε-greedy), `target` (soft-tracked for the
//    double-DQN bootstrap), and `opponent` (frozen-per-episode policy used by seats 1+).
//  - At the start of each episode we pick the opponent's weights — with probability
//    `pPool` from a rolling pool of historical snapshots, otherwise from current online.
//    The opponent is then frozen for the whole episode; this kills the worst kind of
//    self-play non-stationarity (online policy drifting against its own moving target).
//  - Every `snapshotEvery` episodes we copy the online weights into the pool. When the
//    pool exceeds `poolSize` we evict the oldest entry.
//  - All four seats record (s, a, r, s', done) tuples into one shared replay buffer.
//    DQN is off-policy, so opponent transitions are valid training data even though
//    the action comes from the frozen policy.
//  - Decisions are committed only at cell-aligned ticks. Intermediate ticks keep the
//    previous action.
//  - MU-only: trains on 4-player matches from episode 0. The bot is never used in
//    time-trial games, so we don't mimic any TT behavior.
//
// Usage:
//   npx ts-node src/train.ts --episodes=2000 --save=checkpoints/latest.json

import './tfBackend'; // must come first — hijacks the tfjs-node module if TFJS_GPU=1
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { Env } from './env';
import {
    NUM_ACTIONS,
    buildModel,
    exportWeights,
    importWeights,
    softUpdate,
    snapshotWeights,
    obsBatchToTensors,
    flatBatchToTensors,
} from './model';
import { OBS_SIZE } from './observation';
import { Action } from './sim';

interface Args {
    episodes: number;
    save: string;
    bufferSize: number;
    batchSize: number;
    gamma: number;
    learnEvery: number;
    targetTau: number;
    epsStart: number;
    epsEnd: number;
    epsDecaySteps: number;
    learningRate: number;
    saveEvery: number;
    seed: number;
    // Opponent pool knobs.
    poolSize: number;       // max number of historical snapshots kept around
    snapshotEvery: number;  // add online weights to the pool every N episodes
    pPool: number;          // P(opponent uses a pool sample) vs current online
    // 'auto' (default) loads --save if it exists; 'on' errors if not present;
    // 'off' starts fresh and overwrites any existing checkpoint.
    resume: 'auto' | 'on' | 'off';
}

function parseArgs(): Args {
    const a: Args = {
        episodes: 2000,
        save: 'checkpoints/latest.json',
        bufferSize: 100_000,
        batchSize: 128,
        gamma: 0.99,
        learnEvery: 8,
        targetTau: 0.005,
        epsStart: 1.0,
        epsEnd: 0.05,
        // 3M env-steps ≈ 30,000s of in-game time at 10ms ticks (~600 games at
        // ~5000 steps/game). 4-player multi-agent dynamics need a longer
        // exploration tail than a 1v1 game would; this leaves ~70% of a 2000-
        // episode run in pure exploitation rather than ~90%.
        epsDecaySteps: 3_000_000,
        learningRate: 1e-4,
        saveEvery: 50,
        seed: 1,
        poolSize: 10,
        snapshotEvery: 25,
        pPool: 0.5,
        resume: 'auto',
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

// Fixed-size ring replay buffer. Storing Float32Arrays directly is cheaper than tensors
// and lets us copy-slice into one batch tensor at sample time.
class ReplayBuffer {
    cap: number;
    size = 0;
    head = 0;
    obs: Float32Array[];
    next: Float32Array[];
    act: Int32Array;
    rew: Float32Array;
    done: Uint8Array;

    constructor(capacity: number) {
        this.cap = capacity;
        this.obs = new Array(capacity);
        this.next = new Array(capacity);
        this.act = new Int32Array(capacity);
        this.rew = new Float32Array(capacity);
        this.done = new Uint8Array(capacity);
    }

    push(s: Float32Array, a: number, r: number, sNext: Float32Array, done: boolean) {
        this.obs[this.head] = s;
        this.next[this.head] = sNext;
        this.act[this.head] = a;
        this.rew[this.head] = r;
        this.done[this.head] = done ? 1 : 0;
        this.head = (this.head + 1) % this.cap;
        if (this.size < this.cap) this.size++;
    }

    sample(n: number): { s: Float32Array; a: Int32Array; r: Float32Array; sNext: Float32Array; done: Uint8Array } {
        const sBuf = new Float32Array(n * OBS_SIZE);
        const sNextBuf = new Float32Array(n * OBS_SIZE);
        const aBuf = new Int32Array(n);
        const rBuf = new Float32Array(n);
        const dBuf = new Uint8Array(n);
        for (let i = 0; i < n; i++) {
            const idx = Math.floor(Math.random() * this.size);
            sBuf.set(this.obs[idx], i * OBS_SIZE);
            sNextBuf.set(this.next[idx], i * OBS_SIZE);
            aBuf[i] = this.act[idx];
            rBuf[i] = this.rew[idx];
            dBuf[i] = this.done[idx];
        }
        return { s: sBuf, a: aBuf, r: rBuf, sNext: sNextBuf, done: dBuf };
    }
}

// FIFO of frozen weight snapshots. Owns the tensors and disposes them on eviction.
class OpponentPool {
    private entries: { weights: tf.Tensor[]; episode: number }[] = [];
    constructor(public capacity: number) {}

    add(weights: tf.Tensor[], episode: number) {
        this.entries.push({ weights, episode });
        while (this.entries.length > this.capacity) {
            const dropped = this.entries.shift()!;
            dropped.weights.forEach(w => w.dispose());
        }
    }

    sample(): tf.Tensor[] | null {
        if (this.entries.length === 0) return null;
        return this.entries[Math.floor(Math.random() * this.entries.length)].weights;
    }

    size(): number { return this.entries.length; }

    dispose() {
        for (const e of this.entries) e.weights.forEach(w => w.dispose());
        this.entries = [];
    }
}

function epsilon(step: number, a: Args): number {
    const frac = Math.min(1, step / a.epsDecaySteps);
    return a.epsStart + (a.epsEnd - a.epsStart) * frac;
}

function pickAction(qValues: Float32Array, eps: number): Action {
    if (Math.random() < eps) return Math.floor(Math.random() * NUM_ACTIONS) as Action;
    let best = 0;
    let bestV = qValues[0];
    for (let i = 1; i < NUM_ACTIONS; i++) if (qValues[i] > bestV) { bestV = qValues[i]; best = i; }
    return best as Action;
}

function argmax(q: Float32Array): Action {
    let best = 0;
    let bestV = q[0];
    for (let i = 1; i < NUM_ACTIONS; i++) if (q[i] > bestV) { bestV = q[i]; best = i; }
    return best as Action;
}

// Stack N obs into one (spatial, scalar) batch and return Q-vals as N arrays of length 6.
function qBatch(model: tf.LayersModel, obs: Float32Array[]): Float32Array[] {
    return tf.tidy(() => {
        const [spatial, scalar] = obsBatchToTensors(obs);
        const out = model.predict([spatial, scalar]) as tf.Tensor;
        const data = out.dataSync();
        const ret: Float32Array[] = [];
        for (let i = 0; i < obs.length; i++) {
            ret.push((data as Float32Array).slice(i * NUM_ACTIONS, (i + 1) * NUM_ACTIONS));
        }
        return ret;
    });
}

async function train() {
    const args = parseArgs();
    fs.mkdirSync(path.dirname(args.save), { recursive: true });
    console.log('[train] args:', args);

    const online = buildModel();
    const target = buildModel();
    const opponent = buildModel();
    target.setWeights(online.getWeights());
    opponent.setWeights(online.getWeights());

    let globalStep = 0;
    let startEpisode = 0;
    const checkpointExists = fs.existsSync(args.save);
    if (args.resume === 'on' && !checkpointExists) {
        throw new Error(`--resume=on but checkpoint not found at ${args.save}`);
    }
    if (args.resume !== 'off' && checkpointExists) {
        const ts = await importWeights(online, args.save);
        target.setWeights(online.getWeights());
        opponent.setWeights(online.getWeights());
        if (ts) {
            globalStep = ts.globalStep;
            startEpisode = ts.episode;
            console.log(`[train] resumed from ${args.save}: globalStep=${globalStep}, episode=${startEpisode}`);
        } else {
            console.log(`[train] loaded weights from ${args.save} (no training state stored — counters reset to 0)`);
        }
    } else {
        console.log('[train] starting from fresh weights');
    }

    // Adam state is NOT persisted — on warm-start it warms up over ~hundreds of steps.
    // Minor cost vs. the simplicity of skipping optimizer-state serialization.
    const optimizer = tf.train.adam(args.learningRate);
    const buffer = new ReplayBuffer(args.bufferSize);
    const pool = new OpponentPool(args.poolSize);

    let totalGames = 0;

    for (let ep = startEpisode; ep < startEpisode + args.episodes; ep++) {
        const numPlayers = 4;
        const env = new Env({
            numPlayers,
            seed: args.seed + ep,
            maxTimeMs: 120_000,
        });

        // Freeze the opponent for this episode. With prob pPool draw from history;
        // otherwise mirror the current online weights (still frozen for the episode,
        // just no historical diversity).
        let opponentSource = 'online';
        if (pool.size() > 0 && Math.random() < args.pPool) {
            const w = pool.sample()!;
            opponent.setWeights(w);
            opponentSource = 'pool';
        } else {
            opponent.setWeights(online.getWeights());
        }

        let obs = env.initialObs();
        let prevObs: Array<Float32Array | null> = new Array(numPlayers).fill(null);
        let prevAct: Array<Action | null> = new Array(numPlayers).fill(null);
        let pendingReward: Float32Array = new Float32Array(numPlayers);
        let totalReward: Float32Array = new Float32Array(numPlayers);
        let done = false;
        let stepInGame = 0;
        // 120s game cap at 10ms ticks ⇒ 12000 ticks exactly; give a little margin.
        const maxStepsPerGame = 15000;

        while (!done && stepInGame < maxStepsPerGame) {
            const eps = epsilon(globalStep, args);

            // Only seats that are cell-aligned + alive + !knocked need a fresh decision
            // this tick. Cells take ~120–220ms to traverse at 10ms/tick, so on most ticks
            // 0 or 1 seats need a forward — skipping the rest is the dominant perf win.
            let learnerQ: Float32Array | null = null;
            const opponentSeats: number[] = [];
            const opponentObs: Float32Array[] = [];
            for (let i = 0; i < numPlayers; i++) {
                const p = env.sim.players[i];
                if (!env.sim.isAtCell(i) || !p.alive || p.knocked) continue;
                if (i === 0) {
                    learnerQ = qBatch(online, [obs[0]])[0];
                } else {
                    opponentSeats.push(i);
                    opponentObs.push(obs[i]);
                }
            }
            const opponentQs = opponentObs.length > 0 ? qBatch(opponent, opponentObs) : [];

            const actions: Action[] = [];
            let opponentBatchIdx = 0;
            for (let i = 0; i < numPlayers; i++) {
                if (!env.sim.isAtCell(i) || !env.sim.players[i].alive || env.sim.players[i].knocked) {
                    actions.push(env.sim.players[i].pendingAction);
                    continue;
                }
                const q = i === 0 ? learnerQ! : opponentQs[opponentBatchIdx++];
                // Learner explores; opponents play greedy from their frozen policy.
                const a = i === 0 ? pickAction(q, eps) : argmax(q);
                actions.push(a);
                if (prevObs[i] !== null && prevAct[i] !== null) {
                    buffer.push(prevObs[i]!, prevAct[i]!, pendingReward[i], obs[i], false);
                }
                prevObs[i] = obs[i];
                prevAct[i] = a;
                pendingReward[i] = 0;
            }

            const res = env.step(actions);
            for (let i = 0; i < numPlayers; i++) {
                pendingReward[i] += res.rewards[i];
                totalReward[i] += res.rewards[i];
            }
            obs = res.obs;
            done = res.done;

            // Learning step.
            if (buffer.size >= args.batchSize && globalStep % args.learnEvery === 0) {
                const batch = buffer.sample(args.batchSize);
                tf.tidy(() => {
                    const [sSpatial, sScalar] = flatBatchToTensors(batch.s, args.batchSize);
                    const [sNextSpatial, sNextScalar] = flatBatchToTensors(batch.sNext, args.batchSize);
                    const aT = tf.tensor1d(batch.a, 'int32');
                    const rT = tf.tensor1d(batch.r);
                    const dT = tf.tensor1d(Float32Array.from(batch.done));

                    // Double DQN target: a* from online net, value from target net.
                    const qNextOnline = online.predict([sNextSpatial, sNextScalar]) as tf.Tensor;
                    const aStar = qNextOnline.argMax(1);
                    const qNextTarget = target.predict([sNextSpatial, sNextScalar]) as tf.Tensor;
                    const indices = aStar.cast('int32');
                    const oneHot = tf.oneHot(indices, NUM_ACTIONS);
                    const qNextSel = qNextTarget.mul(oneHot).sum(1);
                    const yT = rT.add(qNextSel.mul(args.gamma).mul(tf.scalar(1).sub(dT)));

                    const grads = tf.variableGrads(() => {
                        const q = online.predict([sSpatial, sScalar]) as tf.Tensor;
                        const aOH = tf.oneHot(aT, NUM_ACTIONS);
                        const qSel = q.mul(aOH).sum(1);
                        return tf.losses.huberLoss(yT, qSel) as tf.Scalar;
                    });
                    // Cast around tfjs typing: variableGrads().grads is a NamedTensorMap, but
                    // applyGradients() declares NamedVariableMap. Runtime is fine — they're the
                    // same shape; this just keeps tsc happy under strict mode.
                    optimizer.applyGradients(grads.grads as unknown as Parameters<typeof optimizer.applyGradients>[0]);
                    Object.values(grads.grads).forEach((g: tf.Tensor) => g.dispose());
                });
                softUpdate(online, target, args.targetTau);
            }

            globalStep++;
            stepInGame++;
        }

        // Flush trailing transitions with done=true.
        for (let i = 0; i < numPlayers; i++) {
            if (prevObs[i] !== null && prevAct[i] !== null) {
                buffer.push(prevObs[i]!, prevAct[i]!, pendingReward[i], obs[i], true);
            }
        }

        // Snapshot the current online weights into the pool periodically.
        if (ep > startEpisode && ep % args.snapshotEvery === 0) {
            pool.add(snapshotWeights(online), ep);
        }

        totalGames++;
        const meanR = Array.from(totalReward).reduce((a, b) => a + b, 0) / numPlayers;
        console.log(`[train] ep=${ep} step=${globalStep} eps=${epsilon(globalStep, args).toFixed(3)} meanR=${meanR.toFixed(2)} bufSize=${buffer.size} woodLeft=${env.sim.woodLeft} opp=${opponentSource}(pool=${pool.size()})`);
        if (ep > startEpisode && ep % args.saveEvery === 0) {
            await exportWeights(online, args.save, { globalStep, episode: ep + 1 });
            console.log(`[train] saved → ${args.save}`);
        }
    }

    await exportWeights(online, args.save, { globalStep, episode: startEpisode + args.episodes });
    console.log(`[train] final save → ${args.save}`);
    pool.dispose();
}

train().catch(err => {
    console.error(err);
    process.exit(1);
});
