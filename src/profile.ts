// Micro-benchmark for the per-tick hot path: sim step, encode, forward at the
// batch sizes the training loop uses (1, 3, 4, 128), and a full learn step.
// Single output table for easy CPU-vs-GPU comparison.
//
// Usage:
//   npm run profile           # CPU
//   npm run profile:gpu       # GPU (TFJS_GPU=1)

import './tfBackend';
import * as tf from '@tensorflow/tfjs-node';
import { Env } from './env';
import {
    NUM_ACTIONS,
    buildModel,
    obsBatchToTensors,
    flatBatchToTensors,
} from './model';
import { OBS_SIZE } from './observation';
import { Action } from './sim';

function bench(name: string, iters: number, fn: () => void): { name: string; iters: number; meanMs: number; totalMs: number } {
    const t0 = Date.now();
    for (let i = 0; i < iters; i++) fn();
    const total = Date.now() - t0;
    return { name, iters, meanMs: total / iters, totalMs: total };
}

function fmt(rows: { name: string; iters: number; meanMs: number; totalMs: number }[]): string {
    const w1 = Math.max(...rows.map(r => r.name.length), 'name'.length);
    const lines: string[] = [];
    lines.push(`${'name'.padEnd(w1)} | ${'iters'.padStart(6)} | ${'mean (ms)'.padStart(10)} | ${'total (ms)'.padStart(11)}`);
    lines.push('-'.repeat(w1) + '-+-' + '-'.repeat(6) + '-+-' + '-'.repeat(10) + '-+-' + '-'.repeat(11));
    for (const r of rows) {
        lines.push(`${r.name.padEnd(w1)} | ${String(r.iters).padStart(6)} | ${r.meanMs.toFixed(3).padStart(10)} | ${r.totalMs.toString().padStart(11)}`);
    }
    return lines.join('\n');
}

async function main() {
    const backend = process.env.TFJS_GPU ? 'GPU' : 'CPU';
    console.log(`[profile] backend=${backend} tfjs-version=${tf.version.tfjs}`);

    // Warmup later to avoid JIT / cuDNN autotune polluting timings.
    const online = buildModel();
    const target = buildModel();
    target.setWeights(online.getWeights());
    const optimizer = tf.train.adam(1e-4);

    const env = new Env({ numPlayers: 4, seed: 1, maxTimeMs: 120_000 });
    let obs = env.initialObs();

    // Populate non-trivial state (bombs, etc.) before benchmarking.
    for (let i = 0; i < 50; i++) {
        const acts: Action[] = [0, 0, 0, 0];
        const res = env.step(acts);
        obs = res.obs;
    }

    const obs1 = [obs[0]];
    const obs3 = [obs[1], obs[2], obs[3]];
    const obs4 = [obs[0], obs[1], obs[2], obs[3]];

    // Matches train.ts default batchSize.
    const BATCH = 128;
    const replayFlat = new Float32Array(BATCH * OBS_SIZE);
    for (let i = 0; i < BATCH; i++) replayFlat.set(obs[0], i * OBS_SIZE);
    const aBuf = new Int32Array(BATCH);
    const rBuf = new Float32Array(BATCH);
    const dBuf = new Uint8Array(BATCH);
    for (let i = 0; i < BATCH; i++) { aBuf[i] = i % NUM_ACTIONS; rBuf[i] = 0.1; dBuf[i] = 0; }

    // Shared by warmup + bench so the two paths can't drift.
    const runLearnStep = () => {
        tf.tidy(() => {
            const [sS, sSc] = flatBatchToTensors(replayFlat, BATCH);
            const [sNS, sNSc] = flatBatchToTensors(replayFlat, BATCH);
            const aT = tf.tensor1d(aBuf, 'int32');
            const rT = tf.tensor1d(rBuf);
            const dT = tf.tensor1d(Float32Array.from(dBuf));
            const qNextOnline = online.predict([sNS, sNSc]) as tf.Tensor;
            const aStar = qNextOnline.argMax(1).cast('int32');
            const qNextTarget = target.predict([sNS, sNSc]) as tf.Tensor;
            const oneHot = tf.oneHot(aStar, NUM_ACTIONS);
            const qNextSel = qNextTarget.mul(oneHot).sum(1);
            const yT = rT.add(qNextSel.mul(0.99).mul(tf.scalar(1).sub(dT)));
            const grads = tf.variableGrads(() => {
                const q = online.predict([sS, sSc]) as tf.Tensor;
                const aOH = tf.oneHot(aT, NUM_ACTIONS);
                const qSel = q.mul(aOH).sum(1);
                return tf.losses.huberLoss(yT, qSel) as tf.Scalar;
            });
            optimizer.applyGradients(grads.grads as any);
        });
    };

    console.log('[profile] warming up...');
    const warmup = (n: number, obsList: Float32Array[]) => {
        for (let i = 0; i < n; i++) {
            tf.tidy(() => {
                const [s, sc] = obsBatchToTensors(obsList);
                const out = online.predict([s, sc]) as tf.Tensor;
                out.dataSync();
            });
        }
    };
    warmup(50, obs1);
    warmup(50, obs3);
    warmup(50, obs4);
    for (let i = 0; i < 20; i++) runLearnStep();

    const rows: ReturnType<typeof bench>[] = [];

    rows.push(bench('sim step (no model)', 5000, () => {
        const acts: Action[] = [0, 0, 0, 0];
        env.step(acts);
    }));

    rows.push(bench('encode 1 seat', 5000, () => {
        env.initialObs(); // builds 4 seats but we just want per-seat encode cost
    }));

    const benchForward = (label: string, iters: number, batch: Float32Array[]) => {
        rows.push(bench(label, iters, () => {
            tf.tidy(() => {
                const [s, sc] = obsBatchToTensors(batch);
                const out = online.predict([s, sc]) as tf.Tensor;
                out.dataSync();
            });
        }));
    };
    benchForward('forward batch=1   (learner)',  500, obs1);
    benchForward('forward batch=3   (opponents)', 500, obs3);
    benchForward('forward batch=4   (all seats)', 500, obs4);
    benchForward(`forward batch=${BATCH} (replay)`,   200, Array.from({ length: BATCH }, (_, i) => obs[i % 4]));

    rows.push(bench(`learn step (batch=${BATCH})`, 200, runLearnStep));

    console.log('');
    console.log(`[profile] results (backend=${backend}):`);
    console.log(fmt(rows));
    console.log('');

    // 12000-tick episode projection: sim + 1× batch-1 forward + 1× batch-3 + learn/LEARN_EVERY.
    const fwd1 = rows.find(r => r.name.startsWith('forward batch=1'))!.meanMs;
    const fwd3 = rows.find(r => r.name.startsWith('forward batch=3'))!.meanMs;
    const learn = rows.find(r => r.name.startsWith('learn step'))!.meanMs;
    const sim = rows.find(r => r.name.startsWith('sim step'))!.meanMs;
    const LEARN_EVERY = 8; // matches train.ts default
    const perTickSplit = sim + fwd1 + fwd3 + learn / LEARN_EVERY;
    console.log(`[profile] projection for a 12000-tick episode (sim + fwd1 + fwd3 + learn/${LEARN_EVERY}):`);
    console.log(`           per-tick ≈ ${perTickSplit.toFixed(3)} ms  →  episode ≈ ${(perTickSplit * 12).toFixed(1)} s`);
}

main().catch(e => { console.error(e); process.exit(1); });
