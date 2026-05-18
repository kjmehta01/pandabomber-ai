// Self-play Double-Dueling-DQN with prioritized replay + per-seat opponent pool.
//
// Three CNN models: `online` (learner, ε-greedy), `target` (soft-tracked for the
// double-DQN bootstrap), and `opponents[0..N-2]` (one model per non-learner seat,
// independently re-weighted at episode start from the pool or current online).
// All seats' transitions feed one shared PER buffer (DQN is off-policy).
//
// Opponents play with a small fixed ε so they're not perfectly predictable.
// Pool snapshots are step-keyed (not episode-keyed) so cadence is invariant to
// curriculum-driven episode-length changes. Curriculum: easier 2-player games for
// the first N episodes, then 4. Warmup: pure-random until the buffer has diverse
// exploration. Decisions only commit at cell-aligned ticks; intermediate ticks
// re-play pendingAction.
//
// Usage:
//   npx ts-node src/train.ts --episodes=2000 --save=checkpoints/latest.json

import './tfBackend'; // must come first — hijacks tfjs-node if TFJS_GPU=1
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { Env } from './env';
import {
    NUM_ACTIONS,
    buildModel,
    exportWeights,
    importWeights,
    applyOptimizerState,
    softUpdate,
    snapshotWeights,
    obsBatchToTensors,
    flatBatchToTensors,
} from './model';
import { OBS_SIZE } from './observation';
import { Action } from './sim';
import { PrioritizedReplayBuffer } from './per';

interface Args {
    episodes: number;
    save: string;
    bufferSize: number;
    batchSize: number;
    gamma: number;
    learnEvery: number;
    nStep: number;          // n-step return horizon
    targetTau: number;
    epsStart: number;
    epsEnd: number;
    epsDecaySteps: number;
    epsOpponent: number;    // small fixed ε for frozen opponents
    warmupSteps: number;    // pure-random env steps before learning starts
    learningRate: number;
    saveEvery: number;
    seed: number;
    logEvery: number;
    poolSize: number;
    snapshotEverySteps: number;
    pPool: number;          // per-seat P(opponent samples from pool) vs current online
    curriculumPlayers: number;
    curriculumEpisodes: number;
    fullPlayers: number;
    perAlpha: number;
    perBetaStart: number;
    perBetaEnd: number;
    perBetaSteps: number;
    // auto: load --save if present. on: require it. off: start fresh and overwrite.
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
        // n=5 doesn't span a full bomb fuse (~300 ticks) but it propagates
        // knock-then-kill credit (~600 tick horizon) far faster than 1-step TD.
        nStep: 5,
        targetTau: 0.005,
        epsStart: 1.0,
        epsEnd: 0.05,
        epsDecaySteps: 500_000,
        epsOpponent: 0.05,
        warmupSteps: 10_000,
        learningRate: 1e-4,
        saveEvery: 50,
        seed: 1,
        logEvery: 10,
        poolSize: 10,
        snapshotEverySteps: 250_000,
        pPool: 0.5,
        curriculumPlayers: 2,
        curriculumEpisodes: 500,
        fullPlayers: 4,
        perAlpha: 0.6,
        perBetaStart: 0.4,
        perBetaEnd: 1.0,
        perBetaSteps: 3_000_000,
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

// FIFO of frozen weight snapshots; owns + disposes on eviction.
class OpponentPool {
    private entries: { weights: tf.Tensor[]; step: number }[] = [];
    constructor(public capacity: number) {}

    add(weights: tf.Tensor[], step: number) {
        this.entries.push({ weights, step });
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

// One queued decision: the obs the seat saw, the action picked, and the reward
// accumulated across ticks between this decision and the next.
interface StepEntry {
    obs: Float32Array;
    act: Action;
    reward: number;
}

// Pop queue head and push it to the buffer as an n-step transition.
function pushNStep(buffer: PrioritizedReplayBuffer, queue: StepEntry[], bootstrapObs: Float32Array, done: boolean, nSteps: number, gamma: number) {
    const head = queue[0];
    let nReturn = 0;
    let g = 1;
    for (let k = 0; k < nSteps; k++) {
        nReturn += g * queue[k].reward;
        g *= gamma;
    }
    buffer.push(head.obs, head.act, nReturn, bootstrapObs, done, nSteps);
    queue.shift();
}

function epsilon(step: number, a: Args): number {
    if (step < a.warmupSteps) return 1.0;
    const frac = Math.min(1, (step - a.warmupSteps) / Math.max(1, a.epsDecaySteps));
    return a.epsStart + (a.epsEnd - a.epsStart) * frac;
}

function perBeta(step: number, a: Args): number {
    const frac = Math.min(1, step / Math.max(1, a.perBetaSteps));
    return a.perBetaStart + (a.perBetaEnd - a.perBetaStart) * frac;
}

function pickAction(qValues: Float32Array, eps: number): Action {
    if (Math.random() < eps) return Math.floor(Math.random() * NUM_ACTIONS) as Action;
    // Reservoir tie-break: uniform among argmax candidates so flat Q's don't lock onto action 0.
    let best = 0;
    let bestV = qValues[0];
    let nTies = 1;
    for (let i = 1; i < NUM_ACTIONS; i++) {
        if (qValues[i] > bestV) { bestV = qValues[i]; best = i; nTies = 1; }
        else if (qValues[i] === bestV) { nTies++; if (Math.random() < 1 / nTies) best = i; }
    }
    return best as Action;
}

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

// Rolling-window aggregator for episode diagnostics.
class EpisodeStats {
    private rows: {
        learnerWon: number; stalemate: number; learnerAlive: number;
        woodLearner: number; killsLearner: number; deathsLearner: number; knocksScoredLearner: number;
        powerupsLearner: number; ownBombLearner: number; bombsLearner: number;
        rewardLearner: number; rewardOpp: number;
        steps: number; numPlayers: number;
    }[] = [];

    push(row: EpisodeStats['rows'][number]) {
        this.rows.push(row);
        if (this.rows.length > 200) this.rows.shift();
    }

    summary(n: number) {
        const window = this.rows.slice(-n);
        if (window.length === 0) return null;
        const m = window.length;
        const avg = (sel: (r: typeof window[number]) => number) =>
            window.reduce((a, r) => a + sel(r), 0) / m;
        return {
            n: m,
            winRate: avg(r => r.learnerWon),
            stalemateRate: avg(r => r.stalemate),
            survivalRate: avg(r => r.learnerAlive),
            wood: avg(r => r.woodLearner),
            kills: avg(r => r.killsLearner),
            deaths: avg(r => r.deathsLearner),
            knocks: avg(r => r.knocksScoredLearner),
            powerups: avg(r => r.powerupsLearner),
            ownBomb: avg(r => r.ownBombLearner),
            bombs: avg(r => r.bombsLearner),
            rLearner: avg(r => r.rewardLearner),
            rOpp: avg(r => r.rewardOpp),
            steps: avg(r => r.steps),
        };
    }
}

async function train() {
    const args = parseArgs();
    fs.mkdirSync(path.dirname(args.save), { recursive: true });
    console.log('[train] args:', args);

    const online = buildModel();
    const target = buildModel();
    // One model per non-learner seat; weights swapped per episode + seat.
    const maxOpponents = Math.max(1, args.fullPlayers - 1);
    const opponents: tf.LayersModel[] = [];
    for (let i = 0; i < maxOpponents; i++) {
        const m = buildModel();
        m.setWeights(online.getWeights());
        opponents.push(m);
    }
    target.setWeights(online.getWeights());

    let globalStep = 0;
    let startEpisode = 0;
    let lastSnapshotStep = 0;
    let pendingOptState: Awaited<ReturnType<typeof importWeights>>['optimizerState'] | undefined;

    const checkpointExists = fs.existsSync(args.save);
    if (args.resume === 'on' && !checkpointExists) {
        throw new Error(`--resume=on but checkpoint not found at ${args.save}`);
    }
    let resumed = false;
    if (args.resume !== 'off' && checkpointExists) {
        try {
            const loaded = await importWeights(online, args.save);
            target.setWeights(online.getWeights());
            for (const m of opponents) m.setWeights(online.getWeights());
            if (loaded.trainingState) {
                globalStep = loaded.trainingState.globalStep;
                startEpisode = loaded.trainingState.episode;
                lastSnapshotStep = globalStep;
                console.log(`[train] resumed from ${args.save}: globalStep=${globalStep}, episode=${startEpisode}`);
            } else {
                console.log(`[train] loaded weights from ${args.save} (no training state stored — counters reset to 0)`);
            }
            pendingOptState = loaded.optimizerState;
            if (pendingOptState) console.log(`[train] optimizer state queued for re-apply after first learn step (${pendingOptState.length} slot vars)`);
            resumed = true;
        } catch (err) {
            // --resume=on insists on resume; --resume=auto is best-effort and
            // shouldn't kill an unattended run, so move the bad file aside.
            if (args.resume === 'on') throw err;
            const bakPath = `${args.save}.incompat-${Date.now()}.bak`;
            console.warn(`[train] checkpoint at ${args.save} is incompatible: ${(err as Error).message}`);
            console.warn(`[train] renaming → ${bakPath} and starting fresh`);
            fs.renameSync(args.save, bakPath);
        }
    }
    if (!resumed) console.log('[train] starting from fresh weights');

    const optimizer = tf.train.adam(args.learningRate);
    const buffer = new PrioritizedReplayBuffer(args.bufferSize, args.perAlpha);
    const pool = new OpponentPool(args.poolSize);
    const stats = new EpisodeStats();

    // Reset on each summary log so values are means over the last `logEvery` episodes' learn steps.
    let learnLossAcc = 0, learnTdAcc = 0, learnQAcc = 0, learnGradAcc = 0, learnCount = 0;

    for (let ep = startEpisode; ep < startEpisode + args.episodes; ep++) {
        // Curriculum: easier 2-player games for the first N episodes, then 4-player.
        const numPlayers = ep < startEpisode + args.curriculumEpisodes
            ? args.curriculumPlayers
            : args.fullPlayers;
        const env = new Env({
            numPlayers,
            seed: args.seed + ep,
            maxTimeMs: 60_000,
        });

        // Independently freeze each opponent seat for this episode.
        const opponentSources: string[] = ['learner'];
        for (let i = 1; i < numPlayers; i++) {
            const opp = opponents[i - 1];
            if (pool.size() > 0 && Math.random() < args.pPool) {
                opp.setWeights(pool.sample()!);
                opponentSources.push('pool');
            } else {
                opp.setWeights(online.getWeights());
                opponentSources.push('online');
            }
        }
        const poolOppCount = opponentSources.filter(s => s === 'pool').length;

        let obs = env.initialObs();
        const stepQueue: StepEntry[][] = Array.from({ length: numPlayers }, () => []);
        const pendingReward = new Float32Array(numPlayers);
        const totalReward = new Float32Array(numPlayers);
        let done = false;
        let stepInGame = 0;
        // Safety margin past the 60s @ 10ms = 6000 tick cap.
        const maxStepsPerGame = 8000;

        while (!done && stepInGame < maxStepsPerGame) {
            const eps = epsilon(globalStep, args);

            // Only cell-aligned, alive, non-knocked seats need a forward this tick.
            // Each model holds different weights so we can't batch across seats.
            // Cell alignment hits ~every 12–22 ticks per seat, so this is sparse.
            const decisionSeats: number[] = [];
            for (let i = 0; i < numPlayers; i++) {
                const p = env.sim.players[i];
                if (!env.sim.isAtCell(i) || !p.alive || p.knocked) continue;
                decisionSeats.push(i);
            }

            const qValues = new Map<number, Float32Array>();
            for (const i of decisionSeats) {
                const model = i === 0 ? online : opponents[i - 1];
                qValues.set(i, qBatch(model, [obs[i]])[0]);
            }

            const actions: Action[] = [];
            for (let i = 0; i < numPlayers; i++) {
                if (!env.sim.isAtCell(i) || !env.sim.players[i].alive || env.sim.players[i].knocked) {
                    actions.push(env.sim.players[i].pendingAction);
                    continue;
                }
                const a = pickAction(qValues.get(i)!, i === 0 ? eps : args.epsOpponent);
                actions.push(a);

                // Off-policy: opponent transitions are valid training data too.
                const q_i = stepQueue[i];
                if (q_i.length > 0) q_i[q_i.length - 1].reward = pendingReward[i];
                if (q_i.length >= args.nStep) {
                    pushNStep(buffer, q_i, obs[i], false, args.nStep, args.gamma);
                }
                q_i.push({ obs: obs[i], act: a, reward: 0 });
                pendingReward[i] = 0;
            }

            const res = env.step(actions);
            for (let i = 0; i < numPlayers; i++) {
                pendingReward[i] += res.rewards[i];
                totalReward[i] += res.rewards[i];
            }
            obs = res.obs;
            done = res.done;

            const canLearn = globalStep >= args.warmupSteps
                && buffer.size >= args.batchSize
                && globalStep % args.learnEvery === 0;
            if (canLearn) {
                const beta = perBeta(globalStep, args);
                const batch = buffer.sample(args.batchSize, beta);
                const gammaNBuf = new Float32Array(args.batchSize);
                for (let i = 0; i < args.batchSize; i++) {
                    gammaNBuf[i] = Math.pow(args.gamma, batch.nStep[i]);
                }

                // tf.keep |delta| out of variableGrads' scope so we can read TD
                // errors for PER priorities without a second forward pass on `online`.
                const absTd = new Float32Array(args.batchSize);
                tf.tidy(() => {
                    const [sSpatial, sScalar] = flatBatchToTensors(batch.s, args.batchSize);
                    const [sNextSpatial, sNextScalar] = flatBatchToTensors(batch.sNext, args.batchSize);
                    const aT = tf.tensor1d(batch.a, 'int32');
                    const rT = tf.tensor1d(batch.r);
                    const dT = tf.tensor1d(Float32Array.from(batch.done));
                    const gammaNT = tf.tensor1d(gammaNBuf);
                    const isW = tf.tensor1d(batch.isWeights);

                    // Double-DQN target with n-step return.
                    const qNextOnline = online.predict([sNextSpatial, sNextScalar]) as tf.Tensor;
                    const aStar = qNextOnline.argMax(1);
                    const qNextTarget = target.predict([sNextSpatial, sNextScalar]) as tf.Tensor;
                    const oneHot = tf.oneHot(aStar.cast('int32'), NUM_ACTIONS);
                    const qNextSel = qNextTarget.mul(oneHot).sum(1);
                    const yT = rT.add(qNextSel.mul(gammaNT).mul(tf.scalar(1).sub(dT)));

                    let absDeltaKept: tf.Tensor | undefined;
                    let qSelKept: tf.Tensor | undefined;
                    const grads = tf.variableGrads(() => {
                        const q = online.predict([sSpatial, sScalar]) as tf.Tensor;
                        const aOH = tf.oneHot(aT, NUM_ACTIONS);
                        const qSel = q.mul(aOH).sum(1);
                        qSelKept = tf.keep(qSel.clone());
                        // IS-weighted Huber. Written as 0.5*min(|δ|,1)² + (|δ|−min(|δ|,1))
                        // instead of tf.where(|δ|<1, ...) because tfjs-node has no
                        // registered gradient for Less, which the tape errors on
                        // even when the bool tensor isn't on the gradient path.
                        const delta = qSel.sub(yT);
                        const absDelta = delta.abs();
                        absDeltaKept = tf.keep(absDelta.clone());
                        const clipped = tf.minimum(absDelta, 1);
                        const huber = clipped.square().mul(0.5).add(absDelta.sub(clipped));
                        const weighted = huber.mul(isW);
                        return weighted.mean() as tf.Scalar;
                    });

                    const tdArr = absDeltaKept!.dataSync() as Float32Array;
                    for (let i = 0; i < args.batchSize; i++) absTd[i] = tdArr[i];

                    const lossVal = (grads.value.dataSync() as Float32Array)[0];
                    const tdMean = (absDeltaKept!.mean().dataSync() as Float32Array)[0];
                    const qMean = (qSelKept!.mean().dataSync() as Float32Array)[0];
                    const gradNorm = tf.tidy(() => {
                        const sqSums = Object.values(grads.grads).map(g => (g as tf.Tensor).square().sum());
                        return tf.addN(sqSums).sqrt();
                    });
                    const gradNormVal = (gradNorm.dataSync() as Float32Array)[0];
                    gradNorm.dispose();
                    learnLossAcc += lossVal;
                    learnTdAcc += tdMean;
                    learnQAcc += qMean;
                    learnGradAcc += gradNormVal;
                    learnCount++;

                    optimizer.applyGradients(grads.grads as unknown as Parameters<typeof optimizer.applyGradients>[0]);
                });
                buffer.updatePriorities(batch.indices, absTd);
                softUpdate(online, target, args.targetTau);

                // First learn step allocates Adam's slot variables — apply any
                // pending optimizer state from resume now.
                if (pendingOptState) {
                    await applyOptimizerState(optimizer, pendingOptState);
                    pendingOptState = undefined;
                }
            }

            if (globalStep >= args.warmupSteps && globalStep - lastSnapshotStep >= args.snapshotEverySteps) {
                pool.add(snapshotWeights(online), globalStep);
                lastSnapshotStep = globalStep;
            }

            globalStep++;
            stepInGame++;
        }

        // Drain remaining queue entries as done=true transitions.
        for (let i = 0; i < numPlayers; i++) {
            const q_i = stepQueue[i];
            if (q_i.length === 0) continue;
            q_i[q_i.length - 1].reward = pendingReward[i];
            while (q_i.length > 0) {
                pushNStep(buffer, q_i, obs[i], true, q_i.length, args.gamma);
            }
        }

        const learnerPlayer = env.sim.players[0];
        const aliveCount = env.sim.players.filter(p => p.alive).length;
        const stalemate = env.sim.elapsedMs >= 60_000 && aliveCount > 1;
        const learnerWon = learnerPlayer.alive && aliveCount === 1;
        const ls = learnerPlayer.stats;
        const oppRewardMean = numPlayers > 1
            ? (Array.from(totalReward).slice(1).reduce((a, b) => a + b, 0) / (numPlayers - 1))
            : 0;
        stats.push({
            learnerWon: learnerWon ? 1 : 0,
            stalemate: stalemate ? 1 : 0,
            learnerAlive: learnerPlayer.alive ? 1 : 0,
            woodLearner: ls.woodDestroyed,
            killsLearner: ls.killsScored,
            deathsLearner: learnerPlayer.alive ? 0 : 1,
            knocksScoredLearner: ls.knocksScored,
            powerupsLearner: ls.powerupsCollected,
            ownBombLearner: ls.diedFromOwnBomb,
            bombsLearner: ls.bombsPlaced,
            rewardLearner: totalReward[0],
            rewardOpp: oppRewardMean,
            steps: stepInGame,
            numPlayers,
        });

        const lc = Math.max(1, learnCount);
        const learnLine = learnCount > 0
            ? ` learn[n=${learnCount}]: loss=${(learnLossAcc / lc).toExponential(2)} |td|=${(learnTdAcc / lc).toFixed(3)} q=${(learnQAcc / lc).toFixed(3)} gradNorm=${(learnGradAcc / lc).toFixed(3)}`
            : ' learn[n=0]';
        console.log(
            `[train] ep=${ep} step=${globalStep} eps=${epsilon(globalStep, args).toFixed(3)} ` +
            `np=${numPlayers} rLearner=${totalReward[0].toFixed(2)} rOpp=${oppRewardMean.toFixed(2)} ` +
            `wood=${ls.woodDestroyed} kills=${ls.killsScored} alive=${learnerPlayer.alive ? 1 : 0} ` +
            `bufSize=${buffer.size} opps=${opponentSources.slice(1).join(',')} pool=${pool.size()}/${args.poolSize} (pPool=${poolOppCount})` +
            learnLine
        );
        learnLossAcc = 0; learnTdAcc = 0; learnQAcc = 0; learnGradAcc = 0; learnCount = 0;

        if ((ep + 1) % args.logEvery === 0) {
            const s = stats.summary(args.logEvery)!;
            console.log(
                `[summary] ep=${ep + 1} last${s.n}: winRate=${(s.winRate * 100).toFixed(1)}% ` +
                `stalemate=${(s.stalemateRate * 100).toFixed(1)}% survive=${(s.survivalRate * 100).toFixed(1)}% ` +
                `wood=${s.wood.toFixed(1)} kills=${s.kills.toFixed(2)} knocks=${s.knocks.toFixed(2)} deaths=${s.deaths.toFixed(2)} ` +
                `ownBomb=${s.ownBomb.toFixed(2)} bombs=${s.bombs.toFixed(1)} powerups=${s.powerups.toFixed(1)} ` +
                `rLearner=${s.rLearner.toFixed(2)} rOpp=${s.rOpp.toFixed(2)} steps=${s.steps.toFixed(0)}`
            );
        }

        if (ep > startEpisode && ep % args.saveEvery === 0) {
            await exportWeights(online, args.save, { globalStep, episode: ep + 1 }, optimizer);
            console.log(`[train] saved → ${args.save}`);
        }
    }

    await exportWeights(online, args.save, { globalStep, episode: startEpisode + args.episodes }, optimizer);
    console.log(`[train] final save → ${args.save}`);
    pool.dispose();
}

train().catch(err => {
    console.error(err);
    process.exit(1);
});
