// Network architecture + weight serialization.
//
// Architecture: small CNN over the spatial channels (13 × 19 × 11), with the scalar
// features concatenated into the dense head:
//   Conv 3×3 same  → 32 filters → ReLU
//   Conv 3×3 same  → 32 filters → ReLU
//   Flatten + concat scalars
//   Dense 64 → ReLU
//   Dense 6 (linear Q-values)
//
// The two-input model lets us reshape the spatial slice of the flat observation buffer
// directly into a tensor4d without any transpose (the obs layout is already NHWC).
// bot2.ts implements the same forward pass in pure JS (small conv2d loop + matmuls)
// so the gameserver doesn't need TF.js.
//
// Weight serialization stores each layer's weights as { shape, data } so we can
// round-trip 4D conv kernels and 2D dense kernels through the same code path.

import * as tf from '@tensorflow/tfjs-node';
import {
    OBS_SIZE,
    OBS_SPATIAL_SIZE,
    BOARD_H,
    BOARD_W,
    NUM_CHANNELS,
    NUM_SCALARS,
} from './observation';

export const NUM_ACTIONS = 6;
export const CONV1_FILTERS = 32;
export const CONV2_FILTERS = 32;
export const DENSE_UNITS = 64;
export const KERNEL_SIZE = 3;

const LAYER_NAMES = ['conv1', 'conv2', 'dense1', 'q'] as const;

export function buildModel(): tf.LayersModel {
    const spatialInput = tf.input({ shape: [BOARD_H, BOARD_W, NUM_CHANNELS], name: 'spatial' });
    const scalarInput = tf.input({ shape: [NUM_SCALARS], name: 'scalars' });

    let x: tf.SymbolicTensor = tf.layers.conv2d({
        filters: CONV1_FILTERS, kernelSize: KERNEL_SIZE, padding: 'same',
        activation: 'relu', name: 'conv1',
    }).apply(spatialInput) as tf.SymbolicTensor;
    x = tf.layers.conv2d({
        filters: CONV2_FILTERS, kernelSize: KERNEL_SIZE, padding: 'same',
        activation: 'relu', name: 'conv2',
    }).apply(x) as tf.SymbolicTensor;
    x = tf.layers.flatten().apply(x) as tf.SymbolicTensor;
    const combined = tf.layers.concatenate().apply([x, scalarInput]) as tf.SymbolicTensor;
    let y: tf.SymbolicTensor = tf.layers.dense({
        units: DENSE_UNITS, activation: 'relu', name: 'dense1',
    }).apply(combined) as tf.SymbolicTensor;
    const q = tf.layers.dense({
        units: NUM_ACTIONS, activation: 'linear', name: 'q',
    }).apply(y) as tf.SymbolicTensor;

    return tf.model({ inputs: [spatialInput, scalarInput], outputs: q });
}

// Split a single flat observation into the (spatial, scalar) tensors the model expects.
// Caller is responsible for disposing the returned tensors.
export function obsToTensors(obs: Float32Array): [tf.Tensor4D, tf.Tensor2D] {
    const spatial = tf.tensor4d(obs.subarray(0, OBS_SPATIAL_SIZE), [1, BOARD_H, BOARD_W, NUM_CHANNELS]);
    const scalar = tf.tensor2d(obs.subarray(OBS_SPATIAL_SIZE), [1, NUM_SCALARS]);
    return [spatial, scalar];
}

// Batched version: pack N obs into one pair of tensors.
export function obsBatchToTensors(obs: Float32Array[]): [tf.Tensor4D, tf.Tensor2D] {
    const N = obs.length;
    const spatialBuf = new Float32Array(N * OBS_SPATIAL_SIZE);
    const scalarBuf = new Float32Array(N * NUM_SCALARS);
    for (let i = 0; i < N; i++) {
        spatialBuf.set(obs[i].subarray(0, OBS_SPATIAL_SIZE), i * OBS_SPATIAL_SIZE);
        scalarBuf.set(obs[i].subarray(OBS_SPATIAL_SIZE), i * NUM_SCALARS);
    }
    const spatial = tf.tensor4d(spatialBuf, [N, BOARD_H, BOARD_W, NUM_CHANNELS]);
    const scalar = tf.tensor2d(scalarBuf, [N, NUM_SCALARS]);
    return [spatial, scalar];
}

// Same split, but for a contiguous batched flat buffer (used by the replay-sample path).
export function flatBatchToTensors(flat: Float32Array, batchSize: number): [tf.Tensor4D, tf.Tensor2D] {
    const spatialBuf = new Float32Array(batchSize * OBS_SPATIAL_SIZE);
    const scalarBuf = new Float32Array(batchSize * NUM_SCALARS);
    for (let i = 0; i < batchSize; i++) {
        const off = i * OBS_SIZE;
        spatialBuf.set(flat.subarray(off, off + OBS_SPATIAL_SIZE), i * OBS_SPATIAL_SIZE);
        scalarBuf.set(flat.subarray(off + OBS_SPATIAL_SIZE, off + OBS_SIZE), i * NUM_SCALARS);
    }
    const spatial = tf.tensor4d(spatialBuf, [batchSize, BOARD_H, BOARD_W, NUM_CHANNELS]);
    const scalar = tf.tensor2d(scalarBuf, [batchSize, NUM_SCALARS]);
    return [spatial, scalar];
}

export interface TrainingState {
    // Cross-run training counters so warm-starts pick up epsilon-decay where they left off.
    globalStep: number;
    episode: number;
}

interface SerializedWeight {
    shape: number[];
    data: number[];
}

export interface SerializedModel {
    arch: 'cnn-v1';
    obsSize: number;
    numActions: number;
    boardH: number;
    boardW: number;
    numChannels: number;
    numScalars: number;
    conv1Filters: number;
    conv2Filters: number;
    denseUnits: number;
    kernelSize: number;
    layers: Array<{ name: string; weights: SerializedWeight[] }>;
    trainingState?: TrainingState;
}

export async function exportWeights(model: tf.LayersModel, path: string, trainingState?: TrainingState): Promise<void> {
    const out: SerializedModel = {
        arch: 'cnn-v1',
        obsSize: OBS_SIZE,
        numActions: NUM_ACTIONS,
        boardH: BOARD_H,
        boardW: BOARD_W,
        numChannels: NUM_CHANNELS,
        numScalars: NUM_SCALARS,
        conv1Filters: CONV1_FILTERS,
        conv2Filters: CONV2_FILTERS,
        denseUnits: DENSE_UNITS,
        kernelSize: KERNEL_SIZE,
        layers: [],
        trainingState,
    };
    for (const name of LAYER_NAMES) {
        const layer = model.getLayer(name);
        const ws = layer.getWeights();
        const weights: SerializedWeight[] = [];
        for (const w of ws) {
            const shape = w.shape.slice();
            const data = Array.from(await w.data() as Float32Array);
            weights.push({ shape, data });
        }
        out.layers.push({ name, weights });
    }
    const fs = await import('fs');
    // Atomic write: stage to .tmp then rename, so a crash mid-save can't truncate the
    // existing checkpoint and lose hours of training.
    const tmp = path + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(out));
    fs.renameSync(tmp, path);
}

// Load weights and (optionally) training state from a checkpoint JSON into `model`.
// Throws if the checkpoint's arch/dims don't match the freshly-built model — we'd
// rather fail loudly than silently start from random weights when the user asked
// to warm-start.
export async function importWeights(model: tf.LayersModel, path: string): Promise<TrainingState | undefined> {
    const fs = await import('fs');
    const raw = fs.readFileSync(path, 'utf8');
    const data = JSON.parse(raw) as SerializedModel;
    if (data.arch !== 'cnn-v1') throw new Error(`incompatible arch: ${data.arch}`);
    if (data.obsSize !== OBS_SIZE) throw new Error(`obsSize mismatch: checkpoint=${data.obsSize} model=${OBS_SIZE}`);
    if (data.numActions !== NUM_ACTIONS) throw new Error(`numActions mismatch: checkpoint=${data.numActions} model=${NUM_ACTIONS}`);
    if (data.boardH !== BOARD_H || data.boardW !== BOARD_W || data.numChannels !== NUM_CHANNELS || data.numScalars !== NUM_SCALARS) {
        throw new Error(`board/channel layout mismatch`);
    }
    if (data.conv1Filters !== CONV1_FILTERS || data.conv2Filters !== CONV2_FILTERS || data.denseUnits !== DENSE_UNITS || data.kernelSize !== KERNEL_SIZE) {
        throw new Error(`architecture hyperparameter mismatch`);
    }
    for (const ld of data.layers) {
        const layer = model.getLayer(ld.name);
        const tensors = ld.weights.map(w => tf.tensor(w.data, w.shape));
        layer.setWeights(tensors);
        tensors.forEach(t => t.dispose());
    }
    return data.trainingState;
}

// Copy weights from source to target without rebuilding (target net update).
export function copyWeights(src: tf.LayersModel, dst: tf.LayersModel): void {
    const sw = src.getWeights();
    dst.setWeights(sw);
}

// Soft-update: dst ← τ·src + (1-τ)·dst. Smoother target tracking than hard copies.
export function softUpdate(src: tf.LayersModel, dst: tf.LayersModel, tau: number): void {
    const sw = src.getWeights();
    const dw = dst.getWeights();
    const next = sw.map((s, i) => tf.tidy(() => s.mul(tau).add(dw[i].mul(1 - tau))));
    dst.setWeights(next);
    next.forEach(t => t.dispose());
}

// Snapshot the model's current weights as a detached array (caller owns dispose).
// Used by the opponent pool to keep frozen historical policies around.
export function snapshotWeights(model: tf.LayersModel): tf.Tensor[] {
    return model.getWeights().map(w => w.clone());
}
