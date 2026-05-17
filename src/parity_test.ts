// Parity test: confirms bot2's pure-JS forward pass matches tfjs's predict() bit-for-bit
// (within floating-point tolerance). If this drifts, the trained model won't transfer.
//
// We replicate bot2's conv2d + dense routines here and feed both the model and the
// hand-rolled forward the SAME weights + SAME observation.

import './tfBackend'; // must come first — hijacks the tfjs-node module if TFJS_GPU=1
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { Env } from './env';
import { buildModel, exportWeights, importWeights, obsToTensors, SerializedModel } from './model';
import { OBS_SIZE, OBS_SPATIAL_SIZE, BOARD_H, BOARD_W, NUM_CHANNELS, NUM_SCALARS } from './observation';

// Copies of bot2's forward routines. Keep in sync with backend-gameserver/src/bot2.ts.
function conv2dRelu(input: Float32Array, inH: number, inW: number, inC: number,
                    kernel: Float32Array, kSize: number, outC: number, bias: Float32Array): Float32Array {
    const out = new Float32Array(inH * inW * outC);
    const pad = (kSize - 1) >> 1;
    for (let r = 0; r < inH; r++) {
        for (let c = 0; c < inW; c++) {
            const outBase = (r * inW + c) * outC;
            for (let oc = 0; oc < outC; oc++) out[outBase + oc] = bias[oc];
            for (let kr = 0; kr < kSize; kr++) {
                const ir = r + kr - pad;
                if (ir < 0 || ir >= inH) continue;
                for (let kc = 0; kc < kSize; kc++) {
                    const ic = c + kc - pad;
                    if (ic < 0 || ic >= inW) continue;
                    const inBase = (ir * inW + ic) * inC;
                    const kBase = ((kr * kSize + kc) * inC) * outC;
                    for (let ich = 0; ich < inC; ich++) {
                        const v = input[inBase + ich];
                        if (v === 0) continue;
                        const kOff = kBase + ich * outC;
                        for (let oc = 0; oc < outC; oc++) out[outBase + oc] += v * kernel[kOff + oc];
                    }
                }
            }
            for (let oc = 0; oc < outC; oc++) if (out[outBase + oc] < 0) out[outBase + oc] = 0;
        }
    }
    return out;
}

function dense(input: Float32Array, kernel: Float32Array, bias: Float32Array, outDim: number, relu: boolean): Float32Array {
    const inDim = input.length;
    const out = new Float32Array(outDim);
    out.set(bias);
    for (let i = 0; i < inDim; i++) {
        const v = input[i];
        if (v === 0) continue;
        const off = i * outDim;
        for (let j = 0; j < outDim; j++) out[j] += v * kernel[off + j];
    }
    if (relu) for (let j = 0; j < outDim; j++) if (out[j] < 0) out[j] = 0;
    return out;
}

interface LayerW { kernel: Float32Array; kernelShape: number[]; bias: Float32Array }

function loadLayers(ckpt: SerializedModel): { conv1: LayerW; conv2: LayerW; dense1: LayerW; q: LayerW } {
    const byName = new Map(ckpt.layers.map(l => [l.name, l]));
    const get = (n: string): LayerW => {
        const l = byName.get(n)!;
        return {
            kernel: Float32Array.from(l.weights[0].data),
            kernelShape: l.weights[0].shape,
            bias: Float32Array.from(l.weights[1].data),
        };
    };
    return { conv1: get('conv1'), conv2: get('conv2'), dense1: get('dense1'), q: get('q') };
}

function jsForward(obs: Float32Array, w: ReturnType<typeof loadLayers>, meta: SerializedModel): Float32Array {
    const { boardH, boardW, numChannels, numScalars, conv1Filters, conv2Filters, denseUnits, kernelSize, numActions } = meta;
    const spatial = obs.subarray(0, boardH * boardW * numChannels);
    const scalars = obs.subarray(boardH * boardW * numChannels, boardH * boardW * numChannels + numScalars);
    const a1 = conv2dRelu(spatial, boardH, boardW, numChannels, w.conv1.kernel, kernelSize, conv1Filters, w.conv1.bias);
    const a2 = conv2dRelu(a1, boardH, boardW, conv1Filters, w.conv2.kernel, kernelSize, conv2Filters, w.conv2.bias);
    const flatLen = boardH * boardW * conv2Filters;
    const denseIn = new Float32Array(flatLen + numScalars);
    denseIn.set(a2, 0);
    denseIn.set(scalars, flatLen);
    const h = dense(denseIn, w.dense1.kernel, w.dense1.bias, denseUnits, true);
    return dense(h, w.q.kernel, w.q.bias, numActions, false);
}

async function main() {
    const tmpPath = '/tmp/parity_ckpt.json';
    // Build + save random weights.
    const model = buildModel();
    await exportWeights(model, tmpPath);

    // Load JS-side weights from the saved JSON (same path bot2 takes).
    const ckpt = JSON.parse(fs.readFileSync(tmpPath, 'utf8')) as SerializedModel;
    const weights = loadLayers(ckpt);

    // Build an observation by running a Sim a few ticks then pulling player 0's view.
    const env = new Env({ numPlayers: 4, seed: 42 });
    for (let i = 0; i < 50; i++) env.step([0, 0, 0, 0]);
    const obs = env.initialObs()[0];

    // tfjs prediction.
    const tfQ = tf.tidy(() => {
        const [s, x] = obsToTensors(obs);
        const out = model.predict([s, x]) as tf.Tensor;
        return Array.from(out.dataSync());
    });

    // JS-side prediction.
    const jsQ = Array.from(jsForward(obs, weights, ckpt));

    console.log('tfjs Q:', tfQ.map(v => v.toFixed(6)));
    console.log('js   Q:', jsQ.map(v => v.toFixed(6)));

    let maxAbsErr = 0;
    for (let i = 0; i < tfQ.length; i++) {
        const e = Math.abs(tfQ[i] - jsQ[i]);
        if (e > maxAbsErr) maxAbsErr = e;
    }
    console.log(`max |tf - js| = ${maxAbsErr.toExponential(3)}`);
    if (maxAbsErr > 1e-3) {
        console.error('FAIL: parity error exceeds 1e-3');
        process.exit(1);
    }
    console.log('PASS: bot2 pure-JS forward matches tfjs within tolerance');

    fs.unlinkSync(tmpPath);
}

main().catch(err => { console.error(err); process.exit(1); });
