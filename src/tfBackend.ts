// Backend selector for @tensorflow/tfjs-node vs @tensorflow/tfjs-node-gpu.
//
// Set TFJS_GPU=1 to use the CUDA build. Side-effect-only: when GPU is requested,
// we hijack Node's module cache so any subsequent `import * as tf from
// '@tensorflow/tfjs-node'` returns the GPU module instead. This keeps the import
// sites identical between CPU and GPU runs (and keeps the tf namespace type intact)
// at the cost of one require.cache shim at startup.
//
// Entry points (train.ts, eval.ts, parity_test.ts) must `import './tfBackend';` as
// their FIRST import, before anything else that depends on @tensorflow/tfjs-node.
// Re-imported modules (like model.ts) then pick up the swap automatically.

if (process.env.TFJS_GPU === '1') {
    try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const gpuModule = require('@tensorflow/tfjs-node-gpu');
        const cpuPath = require.resolve('@tensorflow/tfjs-node');
        // Stuff a synthetic module record into the cache so subsequent
        // require('@tensorflow/tfjs-node') returns the GPU build.
        require.cache[cpuPath] = {
            exports: gpuModule,
            id: cpuPath,
            filename: cpuPath,
            loaded: true,
            children: [],
            paths: [],
            // Casts: Node's Module type has more fields than we set, but the resolver
            // only reads `exports` so the rest can stay defaulted.
        } as unknown as NodeJS.Module;
        console.log('[tf] using GPU backend (@tensorflow/tfjs-node-gpu)');
    } catch (err) {
        console.error(`[tf] TFJS_GPU=1 but @tensorflow/tfjs-node-gpu failed to load: ${(err as Error).message}`);
        console.error('[tf] falling back to CPU. Install with `npm install @tensorflow/tfjs-node-gpu` and ensure CUDA 11.8 + cuDNN 8.6 are present.');
    }
}
