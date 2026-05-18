// Backend selector for tfjs-node vs tfjs-node-gpu. Set TFJS_GPU=1 to use CUDA.
//
// We swap the GPU module into require.cache under the CPU module's resolved path,
// so subsequent `import '@tensorflow/tfjs-node'` calls (including transitive ones
// from model.ts) pick up the GPU build without changing any import sites.
//
// Entry points (train.ts, eval.ts, parity_test.ts) must import this FIRST.

if (process.env.TFJS_GPU === '1') {
    try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const gpuModule = require('@tensorflow/tfjs-node-gpu');
        const cpuPath = require.resolve('@tensorflow/tfjs-node');
        require.cache[cpuPath] = {
            exports: gpuModule,
            id: cpuPath,
            filename: cpuPath,
            loaded: true,
            children: [],
            paths: [],
        } as unknown as NodeJS.Module;
        console.log('[tf] using GPU backend (@tensorflow/tfjs-node-gpu)');
    } catch (err) {
        console.error(`[tf] TFJS_GPU=1 but @tensorflow/tfjs-node-gpu failed to load: ${(err as Error).message}`);
        console.error('[tf] falling back to CPU. Install with `npm install @tensorflow/tfjs-node-gpu` and ensure CUDA 11.8 + cuDNN 8.6 are present.');
    }
}
