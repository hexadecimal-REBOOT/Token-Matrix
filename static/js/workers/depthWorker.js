importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js')

let session = null
let inputName = 'pixel_values'
let outputName = 'predicted_depth'
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
    if (session.outputNames?.length) {
        outputName = session.outputNames.includes('predicted_depth')
            ? 'predicted_depth'
            : session.outputNames[0]
    }
    self.postMessage({ type: 'ready', provider })
}

function toCHW(bitmap, target) {
    const canvas = new OffscreenCanvas(target, target)
    const ctx = canvas.getContext('2d')
    const side = Math.min(bitmap.width, bitmap.height)
    const sx = (bitmap.width - side) / 2
    const sy = (bitmap.height - side) / 2
    ctx.drawImage(bitmap, sx, sy, side, side, 0, 0, target, target)
    const rgba = ctx.getImageData(0, 0, target, target).data
    const area = target * target
    const chw = new Float32Array(3 * area)
    for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
        chw[p] = rgba[i] / 255
        chw[p + area] = rgba[i + 1] / 255
        chw[p + 2 * area] = rgba[i + 2] / 255
    }
    return chw
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
            self.postMessage({ type: 'error', error: 'Depth session not ready.' })
            return
        }
        const bitmap = msg.bitmap
        const tensorData = toCHW(bitmap, msg.targetSize)
        bitmap.close()
        const input = new ort.Tensor('float32', tensorData, [
            1,
            3,
            msg.targetSize,
            msg.targetSize
        ])
        try {
            const out = await session.run({ [inputName]: input })
            const tensor = out[outputName] || Object.values(out)[0]
            const dims = tensor.dims || [1, msg.targetSize, msg.targetSize]
            const h = dims[dims.length - 2]
            const w = dims[dims.length - 1]
            const data = tensor.data instanceof Float32Array
                ? tensor.data
                : Float32Array.from(tensor.data)
            self.postMessage(
                {
                    type: 'depth',
                    frameId: msg.frameId,
                    data: data.buffer,
                    size: [h, w],
                    source: 'keyframe'
                },
                [data.buffer]
            )
        } catch (err) {
            self.postMessage({ type: 'error', error: err.message })
        }
    }
}
