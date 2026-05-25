// Dueling-DQN: two 3×3 conv layers (32 filters each) over the 13×19×NUM_CHANNELS spatial
// input, scalars concatenated into a 64-unit dense head, then split into value (1)
// and advantage (NUM_ACTIONS) streams combined as Q = V + (A − mean A).
//
// bot2.ts re-implements this forward pass in pure JS (no TF.js at deploy time);
// parity_test.ts checks they match bit-for-bit.

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

const LAYER_NAMES = ['conv1', 'conv2', 'dense1', 'value', 'advantage'] as const;
// v2: egocentric centered view + 3 new channels (bombPower, ttr, survivability) + safe-neighbors scalar.
export const ARCH_TAG = 'cnn-dueling-v2';

// Dueling combine layer: Q = V + (A − mean A). Stateless.
class DuelingCombine extends tf.layers.Layer {
    static className = 'DuelingCombine';
    constructor(config?: object) { super(config ?? {}); }
    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        const shapes = inputShape as tf.Shape[];
        return shapes[1];
    }
    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        return tf.tidy(() => {
            const [v, a] = inputs as tf.Tensor[];
            const aMean = a.mean(1, true);
            return v.add(a.sub(aMean));
        });
    }
    getClassName() { return DuelingCombine.className; }
}
tf.serialization.registerClass(DuelingCombine as unknown as tf.serialization.SerializableConstructor<tf.serialization.Serializable>);

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
    const y: tf.SymbolicTensor = tf.layers.dense({
        units: DENSE_UNITS, activation: 'relu', name: 'dense1',
    }).apply(combined) as tf.SymbolicTensor;

    const v = tf.layers.dense({ units: 1, activation: 'linear', name: 'value' }).apply(y) as tf.SymbolicTensor;
    const a = tf.layers.dense({ units: NUM_ACTIONS, activation: 'linear', name: 'advantage' }).apply(y) as tf.SymbolicTensor;
    const q = new DuelingCombine().apply([v, a]) as tf.SymbolicTensor;

    return tf.model({ inputs: [spatialInput, scalarInput], outputs: q });
}

// Caller owns dispose for all *ToTensors returns.
export function obsToTensors(obs: Float32Array): [tf.Tensor4D, tf.Tensor2D] {
    const spatial = tf.tensor4d(obs.subarray(0, OBS_SPATIAL_SIZE), [1, BOARD_H, BOARD_W, NUM_CHANNELS]);
    const scalar = tf.tensor2d(obs.subarray(OBS_SPATIAL_SIZE), [1, NUM_SCALARS]);
    return [spatial, scalar];
}

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

export function obsBatchToTensors(obs: Float32Array[]): [tf.Tensor4D, tf.Tensor2D] {
    const flat = new Float32Array(obs.length * OBS_SIZE);
    for (let i = 0; i < obs.length; i++) flat.set(obs[i], i * OBS_SIZE);
    return flatBatchToTensors(flat, obs.length);
}

// Persisted across runs so warm-starts resume epsilon decay at the right step.
export interface TrainingState {
    globalStep: number;
    episode: number;
}

interface SerializedWeight {
    shape: number[];
    data: number[];
}

interface SerializedOptVar {
    name: string;
    shape: number[];
    data: number[];
}

export interface SerializedModel {
    arch: string;
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
    optimizerState?: SerializedOptVar[];
}

export async function exportWeights(
    model: tf.LayersModel,
    path: string,
    trainingState?: TrainingState,
    optimizer?: tf.Optimizer,
): Promise<void> {
    const out: SerializedModel = {
        arch: ARCH_TAG,
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
    if (optimizer) {
        try {
            const ws = await optimizer.getWeights();
            const serialized: SerializedOptVar[] = [];
            for (const nt of ws) {
                serialized.push({
                    name: nt.name,
                    shape: nt.tensor.shape.slice(),
                    data: Array.from(await nt.tensor.data() as Float32Array),
                });
            }
            out.optimizerState = serialized;
        } catch (err) {
            console.warn('[model] failed to serialize optimizer state:', (err as Error).message);
        }
    }
    const fs = await import('fs');
    // Atomic write: stage to .tmp + rename so a mid-save crash can't truncate the checkpoint.
    const tmp = path + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(out));
    fs.renameSync(tmp, path);
}

export interface LoadResult {
    trainingState?: TrainingState;
    optimizerState?: SerializedOptVar[];
}

// Throws on arch/dim mismatch — silent random reinit would be worse than failing loudly.
export async function importWeights(model: tf.LayersModel, path: string): Promise<LoadResult> {
    const fs = await import('fs');
    const raw = fs.readFileSync(path, 'utf8');
    const data = JSON.parse(raw) as SerializedModel;
    if (data.arch !== ARCH_TAG) throw new Error(`incompatible arch: ${data.arch} (expected ${ARCH_TAG})`);
    if (data.obsSize !== OBS_SIZE) throw new Error(`obsSize mismatch: checkpoint=${data.obsSize} model=${OBS_SIZE}`);
    if (data.numActions !== NUM_ACTIONS) throw new Error(`numActions mismatch: checkpoint=${data.numActions} model=${NUM_ACTIONS}`);
    if (data.boardH !== BOARD_H || data.boardW !== BOARD_W || data.numChannels !== NUM_CHANNELS || data.numScalars !== NUM_SCALARS) {
        throw new Error(`board/channel layout mismatch`);
    }
    if (data.conv1Filters !== CONV1_FILTERS || data.conv2Filters !== CONV2_FILTERS || data.denseUnits !== DENSE_UNITS || data.kernelSize !== KERNEL_SIZE) {
        throw new Error(`architecture hyperparameter mismatch`);
    }
    for (const ld of data.layers) {
        if (!LAYER_NAMES.includes(ld.name as typeof LAYER_NAMES[number])) continue;
        const layer = model.getLayer(ld.name);
        const tensors = ld.weights.map(w => tf.tensor(w.data, w.shape));
        layer.setWeights(tensors);
        tensors.forEach(t => t.dispose());
    }
    return { trainingState: data.trainingState, optimizerState: data.optimizerState };
}

// Must be called AFTER the first applyGradients (when Adam's slot variables exist).
// Best-effort: on tfjs version-skew or slot-name shape mismatch, log and continue with fresh momentum.
export async function applyOptimizerState(optimizer: tf.Optimizer, state: SerializedOptVar[]): Promise<boolean> {
    try {
        const named = state.map(s => ({ name: s.name, tensor: tf.tensor(s.data, s.shape) }));
        await optimizer.setWeights(named);
        named.forEach(n => n.tensor.dispose());
        return true;
    } catch (err) {
        console.warn('[model] failed to apply optimizer state, continuing with fresh momentum:', (err as Error).message);
        return false;
    }
}

// dst ← τ·src + (1-τ)·dst.
export function softUpdate(src: tf.LayersModel, dst: tf.LayersModel, tau: number): void {
    const sw = src.getWeights();
    const dw = dst.getWeights();
    const next = sw.map((s, i) => tf.tidy(() => s.mul(tau).add(dw[i].mul(1 - tau))));
    dst.setWeights(next);
    next.forEach(t => t.dispose());
}

// Detached weight clone; caller owns dispose.
export function snapshotWeights(model: tf.LayersModel): tf.Tensor[] {
    return model.getWeights().map(w => w.clone());
}
