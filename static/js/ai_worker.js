/**
 * GOLIATH Edge AI Web Worker (High Precision Person/Intruder Detection)
 * Runs 100% locally from static WASM runtime without network stalls.
 */

// Import local ONNX Runtime Web bundle
importScripts('/static/ort/ort.all.min.js');

let ortSession = null;
let isBusy = false;

const TENSOR_WIDTH = 640;
const TENSOR_HEIGHT = 640;
const TENSOR_AREA = TENSOR_WIDTH * TENSOR_HEIGHT;
const inputFloatArray = new Float32Array(3 * TENSOR_AREA);
const inv255 = 1.0 / 255.0;

const offscreenCanvas = new OffscreenCanvas(TENSOR_WIDTH, TENSOR_HEIGHT);
const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

async function initModel() {
    try {
        if (self.ort && ort.env) {
            ort.env.wasm.wasmPaths = '/static/ort/';
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.simd = true;
        }

        ortSession = await ort.InferenceSession.create('/static/yolov8n.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
        });

        self.postMessage({ type: 'STATUS', status: 'ONLINE', message: 'Local ONNX WASM Session Ready' });
    } catch (err) {
        console.error('[AIWorker] Model load error:', err);
        self.postMessage({ type: 'STATUS', status: 'ERROR', error: String(err) });
    }
}

function calculateIoU(box1, box2) {
    const xA = Math.max(box1[0], box2[0]), yA = Math.max(box1[1], box2[1]);
    const xB = Math.min(box1[2], box2[2]), yB = Math.min(box1[3], box2[3]);
    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    const box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    return interArea / (box1Area + box2Area - interArea);
}

function processPersonNMS(outputData) {
    const numBoxes = 8400;
    let candidates = [];

    // Row 4 in YOLOv8n is PERSON class probability
    for (let i = 0; i < numBoxes; i++) {
        const personConf = outputData[4 * numBoxes + i];

        // High confidence filter for person targets (prevents phantom false positives)
        if (personConf >= 0.45) {
            const cx = outputData[0 * numBoxes + i];
            const cy = outputData[1 * numBoxes + i];
            const w = outputData[2 * numBoxes + i];
            const h = outputData[3 * numBoxes + i];

            // Filter valid bounding box sizes
            if (w > 20 && h > 30 && w < 620 && h < 620) {
                candidates.push({
                    coords: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    conf: personConf,
                    label: "PERSON",
                    cx: cx,
                    cy: cy,
                });
            }
        }
    }

    // Sort by confidence descending
    candidates.sort((a, b) => b.conf - a.conf);

    // Apply strict Non-Maximum Suppression (IoU < 0.38)
    let finalDetections = [];
    while (candidates.length > 0) {
        const current = candidates.shift();
        finalDetections.push(current);
        candidates = candidates.filter(box => calculateIoU(current.coords, box.coords) < 0.38);
        if (finalDetections.length >= 10) break;
    }

    return finalDetections;
}

async function runInference(imageBitmap, timestamp) {
    if (!ortSession || isBusy) {
        if (imageBitmap) imageBitmap.close();
        return;
    }

    isBusy = true;
    const tStart = performance.now();

    try {
        offscreenCtx.drawImage(imageBitmap, 0, 0, TENSOR_WIDTH, TENSOR_HEIGHT);
        imageBitmap.close();

        const imgData = offscreenCtx.getImageData(0, 0, TENSOR_WIDTH, TENSOR_HEIGHT);
        const pixels = imgData.data;

        // Planar transpose to (1, 3, 640, 640)
        let p = 0;
        for (let i = 0; i < TENSOR_AREA; i++) {
            inputFloatArray[i] = pixels[p] * inv255;
            inputFloatArray[i + TENSOR_AREA] = pixels[p + 1] * inv255;
            inputFloatArray[i + 2 * TENSOR_AREA] = pixels[p + 2] * inv255;
            p += 4;
        }

        const tensor = new ort.Tensor('float32', inputFloatArray, [1, 3, TENSOR_WIDTH, TENSOR_HEIGHT]);
        const results = await ortSession.run({ images: tensor });
        const outputData = results[ortSession.outputNames[0]].data;

        const detections = processPersonNMS(outputData);
        const elapsedMs = performance.now() - tStart;

        self.postMessage({
            type: 'DETECTIONS',
            detections: detections,
            inferenceTimeMs: elapsedMs,
            frameTimestamp: timestamp,
        });

    } catch (err) {
        console.error('[AIWorker] Inference error:', err);
    } finally {
        isBusy = false;
    }
}

self.onmessage = async (e) => {
    const data = e.data;
    if (data.type === 'INIT') {
        await initModel();
    } else if (data.type === 'FRAME') {
        await runInference(data.imageBitmap, data.timestamp);
    }
};

initModel();
