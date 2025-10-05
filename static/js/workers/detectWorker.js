importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js')

let session = null
let inputName = 'images'
let outputName = 'output0'
let provider = 'wasm'

async function initSession(buffer, executionProvider) {
    provider = executionProvider || 'wasm'
    if (provider === 'webgpu' && ort.env.webgpu) {
        ort.env.webgpu.powerPreference = 'high-performance'
    } else {
        ort.env.wasm.simd = true
        ort.env.wasm.numThreads = Math.min(4, self.navigator?.hardwareConcurrency || 2)
    }
    session = await ort.InferenceSession.create(buffer, {
        executionProviders: [provider]
    })
    if (session.inputNames?.length) inputName = session.inputNames[0]
    if (session.outputNames?.length) outputName = session.outputNames[0]
    self.postMessage({ type: 'ready', provider })
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x))
}

function decodeYOLO(tensor, pad, origW, origH, confTh = 0.25, nmsTh = 0.45) {
    const data = tensor.data || tensor
    const dims = tensor.dims || []
    let num = 0
    let step = 0
    let off = 0
    if (dims.length === 3 && dims[2] === 84) {
        num = dims[1]
        step = 84
        off = 1
    } else if (dims.length === 3 && dims[1] === 84) {
        num = dims[2]
        step = 1
        off = 84
    } else {
        num = Math.floor(data.length / 84)
        step = 84
        off = 1
    }
    const boxes = []
    const size = pad.size
    for (let i = 0; i < num; i++) {
        const p = i * step
        const cx = data[p + 0 * off]
        const cy = data[p + 1 * off]
        const w = data[p + 2 * off]
        const h = data[p + 3 * off]
        let best = -Infinity
        let cls = -1
        for (let k = 5; k < 84; k++) {
            if (data[p + k * off] > best) {
                best = data[p + k * off]
                cls = k - 5
            }
        }
        const objectness = sigmoid(data[p + 4 * off])
        const conf = objectness * sigmoid(best)
        if (conf < confTh) continue
        const bx = (cx - pad.offx) / pad.scale
        const by = (cy - pad.offy) / pad.scale
        const bw = w / pad.scale
        const bh = h / pad.scale
        const adjX = bx - (size - origW) / 2
        const adjY = by - (size - origH) / 2
        boxes.push({
            x: adjX - bw / 2,
            y: adjY - bh / 2,
            w: bw,
            h: bh,
            conf,
            cls
        })
    }
    boxes.sort((a, b) => b.conf - a.conf)
    const out = []
    for (const b of boxes) {
        let keep = true
        for (const o of out) {
            const x1 = Math.max(b.x, o.x)
            const y1 = Math.max(b.y, o.y)
            const x2 = Math.min(b.x + b.w, o.x + o.w)
            const y2 = Math.min(b.y + b.h, o.y + o.h)
            const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
            const iou =
                inter / (b.w * b.h + o.w * o.h - inter + 1e-9)
            if (iou > nmsTh) {
                keep = false
                break
            }
        }
        if (keep) out.push(b)
    }
    return out
}

self.onmessage = async (e) => {
    const msg = e.data
    if (msg.type === 'init') {
        try {
            await initSession(msg.buffer, msg.executionProvider)
        } catch (err) {
            self.postMessage({ type: 'error', error: err.message })
        }
        return
    }
    if (msg.type === 'run') {
        if (!session) {
            self.postMessage({ type: 'error', error: 'Detection session not ready.' })
            return
        }
        const tensorData = new Float32Array(msg.tensor)
        const input = new ort.Tensor('float32', tensorData, [
            1,
            3,
            msg.pad.size,
            msg.pad.size
        ])
        try {
            const out = await session.run({ [inputName]: input })
            const tensor = out[outputName] || Object.values(out)[0]
            const dets = decodeYOLO(tensor, msg.pad, msg.origWidth, msg.origHeight)
            self.postMessage({
                type: 'detections',
                frameId: msg.frameId,
                detections: dets,
                source: 'keyframe'
            })
        } catch (err) {
            self.postMessage({ type: 'error', error: err.message })
        }
    }
}
