import { CapturePipeline } from './pipeline.js'

const $ = (id) => document.getElementById(id)
const logEl = $('log')
const JSZip = window.JSZip
const ort = window.ort

function log(message, type = 'info') {
    const icon =
        type === 'error'
            ? '❌'
            : type === 'ok'
            ? '✅'
            : type === 'warn'
            ? '⚠️'
            : 'ℹ️'
    logEl.textContent += `\n${icon} ${message}`
    logEl.scrollTop = logEl.scrollHeight
}

function chip(el, text, ok = null) {
    el.textContent = text
    el.classList.remove('ok', 'bad')
    if (ok === true) el.classList.add('ok')
    if (ok === false) el.classList.add('bad')
}

function showLoading() {
    $('loading').style.display = 'flex'
}

function hideLoading() {
    $('loading').style.display = 'none'
}

const canvasPool = new Map()
function acquireCanvas(key, w, h) {
    const maxSize = 4096
    const width = Math.min(maxSize, w)
    const height = Math.min(maxSize, h)
    let entry = canvasPool.get(key)
    if (!entry) {
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d', { willReadFrequently: true })
        entry = { canvas, ctx, width: 0, height: 0 }
        canvasPool.set(key, entry)
    }
    if (entry.width !== width || entry.height !== height) {
        entry.canvas.width = width
        entry.canvas.height = height
        entry.width = width
        entry.height = height
    } else {
        entry.ctx.clearRect(0, 0, width, height)
    }
    return entry
}

const float32Pool = new Map()
function acquireFloat32(key, length) {
    let arr = float32Pool.get(key)
    if (!arr || arr.length !== length) {
        arr = new Float32Array(length)
        float32Pool.set(key, arr)
    }
    return arr
}

const uint32Pool = new Map()
function acquireUint32(key, length) {
    let arr = uint32Pool.get(key)
    if (!arr || arr.length !== length) {
        arr = new Uint32Array(length)
        uint32Pool.set(key, arr)
    } else {
        arr.fill(0)
    }
    return arr
}

function depthRangeApprox(values, lower = 0.05, upper = 0.95) {
    let min = Infinity
    let max = -Infinity
    let valid = 0
    for (let i = 0; i < values.length; i++) {
        const v = values[i]
        if (!Number.isFinite(v)) continue
        if (v < min) min = v
        if (v > max) max = v
        valid++
    }
    if (!valid || !(max > min)) {
        const base = Number.isFinite(min) ? min : 0
        return { low: base, high: base + 1 }
    }
    const bins = acquireUint32('depth_hist', 512)
    const range = max - min
    for (let i = 0; i < values.length; i++) {
        const v = values[i]
        if (!Number.isFinite(v)) continue
        const norm = (v - min) / range
        const idx = Math.min(511, Math.max(0, Math.floor(norm * 511)))
        bins[idx]++
    }
    const targetLow = Math.floor(lower * (valid - 1))
    const targetHigh = Math.floor(upper * (valid - 1))
    let cumulative = 0
    let low = min
    let high = max
    let lowFound = false
    let highFound = false
    for (let i = 0; i < bins.length; i++) {
        cumulative += bins[i]
        if (!lowFound && cumulative > targetLow) {
            low = min + (i / 511) * range
            lowFound = true
        }
        if (!highFound && cumulative > targetHigh) {
            high = min + (i / 511) * range
            highFound = true
            break
        }
    }
    if (!(high > low)) high = low + 1
    return { low, high }
}

const depthPreviewIndexCache = new Map()
function getDepthPreviewIndices(srcW, srcH, dstW, dstH) {
    const key = `${srcW}x${srcH}->${dstW}x${dstH}`
    let entry = depthPreviewIndexCache.get(key)
    if (!entry) {
        const xIdx = new Uint16Array(dstW)
        const yIdx = new Uint16Array(dstH)
        for (let X = 0; X < dstW; X++) {
            xIdx[X] = Math.min(srcW - 1, ((X * srcW) / dstW) | 0)
        }
        for (let Y = 0; Y < dstH; Y++) {
            yIdx[Y] = Math.min(srcH - 1, ((Y * srcH) / dstH) | 0)
        }
        entry = { xIdx, yIdx }
        depthPreviewIndexCache.set(key, entry)
    }
    return entry
}

function depthPreviewFromF32(dLow, wLow, hLow, W, H) {
    const entry = acquireCanvas('depth_preview', W, H)
    const { canvas, ctx } = entry
    if (!entry.depthImage || entry.depthImage.width !== W || entry.depthImage.height !== H) {
        entry.depthImage = new ImageData(W, H)
    }
    const imgData = entry.depthImage
    const data = imgData.data
    const { low, high } = depthRangeApprox(dLow)
    const inv = high > low ? 1 / (high - low) : 1
    const { xIdx, yIdx } = getDepthPreviewIndices(wLow, hLow, W, H)
    let ptr = 0
    for (let Y = 0; Y < H; Y++) {
        const base = yIdx[Y] * wLow
        for (let X = 0; X < W; X++) {
            const raw = dLow[base + xIdx[X]]
            const norm = Math.min(1, Math.max(0, Number.isFinite(raw) ? (raw - low) * inv : 0))
            const r = Math.round(255 * Math.max(0, Math.min(1, -0.5 + 2.8 * norm)))
            const g = Math.round(255 * Math.max(0, Math.min(1, -0.1 + 2.5 * norm)))
            const b = Math.round(255 * (1 - 0.9 * norm))
            data[ptr++] = r
            data[ptr++] = g
            data[ptr++] = b
            data[ptr++] = 255
        }
    }
    ctx.putImageData(imgData, 0, 0)
    return canvas
}

function scaleIntrinsicsTo({ K, fromW, fromH, toW, toH }) {
    const sx = toW / fromW
    const sy = toH / fromH
    return {
        fx: K.fx * sx,
        fy: K.fy * sy,
        cx: K.cx * sx,
        cy: K.cy * sy,
        skew: K.skew || 0
    }
}

function backprojectToPointCloudFloat32({ depthF32, H, W, Kd, rgbCtx = null }) {
    const { fx, fy, cx, cy } = Kd
    const N = H * W
    const xyz = new Float32Array(N * 3)
    let colorsU8 = null
    if (rgbCtx) {
        const id = rgbCtx.getImageData(0, 0, W, H).data
        colorsU8 = new Uint8Array(N * 3)
        for (let i = 0, j = 0; i < id.length; i += 4, j += 3) {
            colorsU8[j] = id[i]
            colorsU8[j + 1] = id[i + 1]
            colorsU8[j + 2] = id[i + 2]
        }
    }
    let c = 0
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const Z = depthF32[y * W + x]
            if (Z > 0 && Number.isFinite(Z)) {
                xyz[c++] = ((x - cx) * Z) / fx
                xyz[c++] = ((y - cy) * Z) / fy
                xyz[c++] = Z
            } else {
                xyz[c++] = NaN
                xyz[c++] = NaN
                xyz[c++] = NaN
            }
        }
    }
    return { xyz, colorsU8 }
}

const __wmSeed = 'TokenMatrix::Confidential::2024'
const __wmTag = (() => {
    let hex = ''
    for (let i = 0; i < __wmSeed.length; i++) {
        hex += __wmSeed.charCodeAt(i).toString(16).padStart(2, '0')
    }
    return hex
})()
function __applyWatermark(target) {
    if (!target.meta) target.meta = {}
    if (!target.meta.__tmx) target.meta.__tmx = __wmTag
    return target
}
function __watermarkProvenance(prov) {
    return { ...prov, watermark: __wmTag }
}

let fovSession = null
async function predictFovDeg(sessionOrNull, imageBitmap, W, H) {
    let canvas
    let ctx
    try {
        if (sessionOrNull) {
            const size = 224
            canvas = document.createElement('canvas')
            canvas.width = size
            canvas.height = size
            ctx = canvas.getContext('2d')
            ctx.drawImage(imageBitmap, 0, 0, W, H, 0, 0, size, size)
            const rgba = ctx.getImageData(0, 0, size, size).data
            const area = size * size
            const chw = new Float32Array(3 * area)
            for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
                chw[p] = rgba[i] / 255
                chw[p + area] = rgba[i + 1] / 255
                chw[p + 2 * area] = rgba[i + 2] / 255
            }
            const input = new ort.Tensor('float32', chw, [1, 3, size, size])
            const out = await sessionOrNull.run({
                [sessionOrNull.inputNames[0]]: input
            })
            const y = out[sessionOrNull.outputNames[0]] || Object.values(out)[0]
            const fov = Array.isArray(y.data) ? y.data[0] : y.data[0]
            return Math.max(20, Math.min(120, fov))
        }
        canvas = document.createElement('canvas')
        canvas.width = 256
        canvas.height = 256
        ctx = canvas.getContext('2d')
        ctx.drawImage(imageBitmap, 0, 0, W, H, 0, 0, 256, 256)
        const id = ctx.getImageData(0, 0, 256, 256)
        let E = 0
        for (let y = 1; y < 255; y++) {
            for (let x = 1; x < 255; x++) {
                const i = (y * 256 + x) * 4
                const ix =
                    id.data[i] -
                    id.data[i - 4] +
                    (id.data[i + 1] - id.data[i - 3]) +
                    (id.data[i + 2] - id.data[i - 2])
                const iy =
                    id.data[i] -
                    id.data[i - 1024] +
                    (id.data[i + 1] - id.data[i - 1023]) +
                    (id.data[i + 2] - id.data[i - 1022])
                E += Math.hypot(ix, iy)
            }
        }
        E /= 255 * 256 * 256 * 3
        return Math.max(25, Math.min(110, 50 + 25 * Math.min(1, E) - 10))
    } finally {
        if (canvas) {
            canvas.width = 1
            canvas.height = 1
        }
    }
}

function createIntrinsicsLadder({ fovOnnxSession = null } = {}) {
    const deg2rad = (d) => (d * Math.PI) / 180
    const clamp = (x, a, b) => Math.max(a, Math.min(b, x))
    const makeK = ({ fx, fy, cx, cy, skew = 0 }) => ({ fx, fy, cx, cy, skew })
    const fuse1D = (arr) => {
        const w = arr.reduce((s, c) => s + 1 / (c.sigma || 1e-6) ** 2, 0)
        const mu =
            arr.reduce((s, c) => s + c.mu / (c.sigma || 1e-6) ** 2, 0) /
            (w || 1)
        return { mu, sigma: Math.sqrt(1 / (w || 1)) }
    }
    const score = (sigRel, ref = 0.07) => clamp(1 / (1 + sigRel / ref), 0, 1)

    function tierDefault({ W, H }) {
        const fovx = 63
        const fx = (0.5 * W) / Math.tan(0.5 * deg2rad(fovx))
        const fy = fx
        const cx = W / 2
        const cy = H / 2
        const sRel = 0.2
        return {
            rung: 'default',
            K: makeK({ fx, fy, cx, cy, skew: 0 }),
            sigma: { fx: sRel * fx, fy: sRel * fy, cx: 8, cy: 8, skew: 2 },
            confidence: score(sRel),
            details: { fovx }
        }
    }

    async function tierFOV({ imageBitmap, W, H }) {
        if (!fovOnnxSession) return null
        try {
            const fovx = await predictFovDeg(fovOnnxSession, imageBitmap, W, H)
            const fx = (0.5 * W) / Math.tan(0.5 * deg2rad(fovx))
            const fy = fx
            const cx = W / 2
            const cy = H / 2
            const sRel = 0.1
            return {
                rung: 'fov_nn',
                K: makeK({ fx, fy, cx, cy, skew: 0 }),
                sigma: { fx: sRel * fx, fy: sRel * fy, cx: 6, cy: 6, skew: 1.5 },
                confidence: score(sRel),
                details: { fovxDeg: fovx }
            }
        } catch (err) {
            console.warn('FOV predictor failed', err)
            return null
        }
    }

    async function estimate({ imageBitmap, width, height }) {
        const W = width
        const H = height
        const tried = []
        const nn = await tierFOV({ imageBitmap, W, H })
        if (nn) tried.push(nn)
        if (!tried.length) tried.push(tierDefault({ W, H }))
        tried.sort((a, b) => b.confidence - a.confidence)
        const best = tried[0]
        const fxF = fuse1D(tried.map((t) => ({ mu: t.K.fx, sigma: t.sigma.fx })))
        const fyF = fuse1D(tried.map((t) => ({ mu: t.K.fy, sigma: t.sigma.fy })))
        const fusedK = makeK({
            fx: fxF.mu,
            fy: fyF.mu,
            cx: best.K.cx,
            cy: best.K.cy,
            skew: best.K.skew
        })
        const rRel = Math.max(
            fxF.sigma / Math.max(1e-6, fxF.mu),
            fyF.sigma / Math.max(1e-6, fyF.mu)
        )
        return {
            K: fusedK,
            model: 'pinhole_browndecenter(0)',
            source: {
                rung: best.rung,
                confidence: best.confidence,
                tried: tried.map((t) => ({
                    rung: t.rung,
                    confidence: t.confidence,
                    K: t.K,
                    sigma: t.sigma,
                    details: t.details
                }))
            },
            confidence: 1 / (1 + rRel / 0.07),
            qc: {
                method: 'precision-weighted fusion',
                fx_sigma: fxF.sigma,
                fy_sigma: fyF.sigma
            }
        }
    }

    return { estimate }
}

let useEP = 'auto'
$('epSel').onchange = () => (useEP = $('epSel').value)

async function pickEP() {
    let want =
        useEP === 'auto' ? (navigator.gpu ? 'webgpu' : 'wasm') : useEP
    if (want === 'webgpu') {
        ort.env.webgpu.powerPreference = 'high-performance'
    } else {
        ort.env.wasm.simd = true
        ort.env.wasm.numThreads = Math.min(
            4,
            navigator.hardwareConcurrency || 2
        )
    }
    chip($('s_ep'), `EP: ${want.toUpperCase()}`, true)
    return want
}

$('bench').onclick = async (e) => {
    e.preventDefault()
    await pickEP()
    log('Execution provider set.', 'ok')
}

const video = $('video')
const view = $('view')
const vctx = view.getContext('2d', { willReadFrequently: true })
let stream = null
let running = false
let captureHandle = null
let pendingOverlay = null
let latestFrameId = -1
let lastFrameTs = performance.now()
let fpsCounter = 0
let currentIntrinsics = null
let lastFrameResult = null
let pipelineReady = false
let depthModelReady = false
let yoloModelReady = false

const pipeline = new CapturePipeline({
    log: (msg) => log(msg, 'info'),
    onDepth: (frame) => handleDepth(frame),
    onDetections: (frame) => handleDetections(frame),
    onFrameReady: (frame) => handleFrameComplete(frame),
    onFlow: () => {},
    requestIntrinsics: () => currentIntrinsics
})

pipeline.setDepthInputSize(parseInt($('depthSize').value, 10))
$('depthSize').addEventListener('change', (e) => {
    const size = parseInt(e.target.value, 10)
    pipeline.setDepthInputSize(size)
})

function updateFpsDisplay() {
    const now = performance.now()
    if (now - lastFrameTs >= 1000) {
        const fps = fpsCounter / ((now - lastFrameTs) / 1000)
        chip($('s_fps'), `FPS: ${fps.toFixed(1)}`, fps >= 15)
        lastFrameTs = now
        fpsCounter = 0
    }
}

async function ensureIntrinsics() {
    if (currentIntrinsics) return currentIntrinsics
    const bmp = await createImageBitmap(view)
    const ladder = createIntrinsicsLadder({
        fovOnnxSession: fovSession
    })
    currentIntrinsics = await ladder.estimate({
        imageBitmap: bmp,
        width: view.width,
        height: view.height
    })
    chip(
        $('s_intr'),
        `INTR: ${currentIntrinsics.source.rung.toUpperCase()} (${(
            currentIntrinsics.confidence * 100
        ).toFixed(0)}%)`,
        currentIntrinsics.confidence >= 0.5
    )
    return currentIntrinsics
}

function drawOverlay() {
    if (!pendingOverlay) return
    const { depthCanvas, boxes } = pendingOverlay
    if (depthCanvas) {
        vctx.save()
        vctx.globalAlpha = 0.35
        vctx.drawImage(depthCanvas, 0, 0, view.width, view.height)
        vctx.restore()
    }
    if (boxes && boxes.length) {
        vctx.save()
        vctx.strokeStyle = '#38f2a5'
        vctx.lineWidth = 2
        vctx.fillStyle = 'rgba(0,0,0,.65)'
        vctx.font = 'bold 12px ui-monospace'
        for (const b of boxes) {
            vctx.strokeRect(b.x, b.y, b.w, b.h)
            const label = `id${b.id} ${(b.conf * 100).toFixed(1)}%`
            const tw = vctx.measureText(label).width + 8
            const th = 18
            vctx.fillRect(b.x, Math.max(0, b.y - th), tw, th)
            vctx.fillStyle = '#38f2a5'
            vctx.fillText(label, b.x + 4, b.y - 5)
            vctx.fillStyle = 'rgba(0,0,0,.65)'
        }
        vctx.restore()
    }
}

async function captureLoop(now, metadata) {
    if (!running) return
    fpsCounter++
    updateFpsDisplay()
    const width = video.videoWidth
    const height = video.videoHeight
    if (width && height) {
        if (view.width !== width || view.height !== height) {
            view.width = width
            view.height = height
        }
        vctx.drawImage(video, 0, 0, width, height)
        drawOverlay()
        const bitmap = await createImageBitmap(video)
        if (!pipelineReady) {
            await pipeline.init()
            pipelineReady = true
        }
        pipeline.enqueueFrame(bitmap, width, height, metadata?.mediaTime || performance.now())
    }
    captureHandle = video.requestVideoFrameCallback(captureLoop)
}

async function startCamera() {
    try {
        if (stream) stream.getTracks().forEach((t) => t.stop())
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        })
        video.srcObject = stream
        await video.play()
        running = true
        pipelineReady = false
        pipeline.start()
        captureHandle = video.requestVideoFrameCallback(captureLoop)
        log(`Camera started: ${video.videoWidth}×${video.videoHeight}`, 'ok')
    } catch (err) {
        log('Camera error: ' + err.message, 'error')
    }
}

function stopCamera() {
    running = false
    if (captureHandle) {
        video.cancelVideoFrameCallback(captureHandle)
        captureHandle = null
    }
    if (stream) {
        stream.getTracks().forEach((t) => t.stop())
        stream = null
    }
    pipeline.stop()
    pipelineReady = false
}

$('startCam').onclick = async (e) => {
    e.preventDefault()
    if (!depthModelReady) {
        log('Load a depth model first.', 'warn')
        return
    }
    await ensureIntrinsics()
    await startCamera()
}

$('stopCam').onclick = (e) => {
    e.preventDefault()
    stopCamera()
}

async function handleDepth(frame) {
    if (!frame.depth || !frame.depthSize) return
    if (frame.id >= latestFrameId) {
        const [h, w] = frame.depthSize
        const overlay = depthPreviewFromF32(frame.depth, w, h, view.width, view.height)
        pendingOverlay = pendingOverlay || {}
        pendingOverlay.depthCanvas = overlay
        pendingOverlay.frameId = frame.id
    }
}

function handleDetections(frame) {
    if (frame.id >= latestFrameId) {
        pendingOverlay = pendingOverlay || {}
        pendingOverlay.boxes = frame.detections || []
        pendingOverlay.frameId = frame.id
    }
}

let recording = false
let recZip = null
let recFrames = 0
let recDepthFrames = []
let recDepthKeyframes = []
let recDetectionKeyframes = []
let recYOLO = []
let recRGB = []
let recIntrByFrame = []
let recPointXYZ = []
let recPointRGB = []
let recConfig = null
let recRawCount = 0

async function handleFrameComplete(frame) {
    latestFrameId = Math.max(latestFrameId, frame.id)
    lastFrameResult = frame
    if (recording) {
        recRawCount++
        const stride = Math.max(1, Number($('frameStride').value) || 1)
        if ((recRawCount - 1) % stride !== 0) {
            return
        }
        const metersPerUnit = Number($('scaleMeters').value) || 1.0
        const depthCopy = new Float32Array(frame.depth)
        recDepthFrames.push({
            data: depthCopy,
            size: frame.depthSize,
            source: frame.depthSource || 'propagated',
            metersPerUnit
        })
        if ((frame.depthSource || 'propagated') === 'keyframe') {
            recDepthKeyframes.push(recDepthFrames.length - 1)
        }
        const dets = (frame.detections || []).map((d) => ({ ...d }))
        recYOLO.push(dets)
        if ((frame.detSource || 'propagated') === 'keyframe') {
            recDetectionKeyframes.push(recYOLO.length - 1)
        }
        const copy = await new Promise((resolve) =>
            view.toBlob((blob) => resolve(blob), 'image/jpeg', 0.85)
        )
        if (copy) recRGB.push(copy)
        const intr = currentIntrinsics || (await ensureIntrinsics())
        recIntrByFrame.push({ K: intr.K, depthSize: frame.depthSize })
        if (recConfig && recConfig.retention === 'FULL') {
            const [h, w] = frame.depthSize
            const Kd = scaleIntrinsicsTo({
                K: intr.K,
                fromW: view.width,
                fromH: view.height,
                toW: w,
                toH: h
            })
            const canvas = document.createElement('canvas')
            canvas.width = w
            canvas.height = h
            const ctx = canvas.getContext('2d')
            ctx.drawImage(view, 0, 0, view.width, view.height, 0, 0, w, h)
            const { xyz, colorsU8 } = backprojectToPointCloudFloat32({
                depthF32: depthCopy,
                H: h,
                W: w,
                Kd,
                rgbCtx: ctx
            })
            recPointXYZ.push(xyz)
            recPointRGB.push(colorsU8)
        }
        recFrames++
        chip($('s_rec'), `REC: ${recFrames} frames`, true)
    }
}

$('loadDepth').onclick = () => $('depthFile').click()
$('depthFile').onchange = async (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    const buffer = await f.arrayBuffer()
    const ep = await pickEP()
    await pipeline.loadDepthModel(buffer, { executionProvider: ep })
    chip($('s_depth'), `DEPTH: ${f.name}`, true)
    $('startRec').disabled = false
    log(`Depth model loaded: ${f.name}`, 'ok')
    depthModelReady = true
}

$('loadYolo').onclick = () => $('yoloFile').click()
$('yoloFile').onchange = async (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    const buffer = await f.arrayBuffer()
    const ep = await pickEP()
    await pipeline.loadDetModel(buffer, { executionProvider: ep })
    chip($('s_yolo'), `YOLO: ${f.name}`, true)
    log(`YOLO model loaded: ${f.name}`, 'ok')
    yoloModelReady = true
}

$('loadFov').onclick = () => $('fovFile').click()
$('fovFile').onchange = async (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    try {
        fovSession = await ort.InferenceSession.create(await f.arrayBuffer(), {
            executionProviders: [navigator.gpu ? 'webgpu' : 'wasm']
        })
        chip($('s_fov'), `FOV-NN: ${f.name}`, true)
        log('FOV model loaded.', 'ok')
    } catch (err) {
        chip($('s_fov'), 'FOV-NN: load failed', false)
        log(err.message, 'error')
    }
}

let currentImage = null
$('imgFiles').onchange = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    const img = new Image()
    img.onload = () => {
        currentImage = img
        const maxL = 1280
        const s = Math.min(1, maxL / Math.max(img.naturalWidth, img.naturalHeight))
        const w = Math.round(img.naturalWidth * s)
        const h = Math.round(img.naturalHeight * s)
        view.width = w
        view.height = h
        vctx.drawImage(img, 0, 0, w, h)
        log(`Loaded ${f.name} (${img.naturalWidth}×${img.naturalHeight})`, 'ok')
    }
    img.onerror = () => log(`Failed to load image: ${f.name}`, 'error')
    img.src = URL.createObjectURL(f)
}
$('drop').ondragover = (e) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
}
$('drop').ondrop = (e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (!f || !f.type.startsWith('image')) return
    const img = new Image()
    img.onload = () => {
        currentImage = img
        const maxL = 1280
        const s = Math.min(1, maxL / Math.max(img.naturalWidth, img.naturalHeight))
        const w = Math.round(img.naturalWidth * s)
        const h = Math.round(img.naturalHeight * s)
        view.width = w
        view.height = h
        vctx.drawImage(img, 0, 0, w, h)
        log(`Dropped ${f.name} (${img.naturalWidth}×${img.naturalHeight})`, 'ok')
    }
    img.onerror = () => log(`Failed to load dropped image: ${f.name}`, 'error')
    img.src = URL.createObjectURL(f)
}
$('drop').onclick = () => $('imgFiles').click()

$('processOne').onclick = async (e) => {
    e.preventDefault()
    if (!currentImage) {
        log('Pick an image first.', 'warn')
        return
    }
    if (!depthModelReady) {
        log('Load a depth model first.', 'warn')
        return
    }
    await pipeline.init()
    showLoading()
    try {
        const bmp = await createImageBitmap(currentImage)
        const frame = await pipeline.processImage(bmp, view.width, view.height)
        await ensureIntrinsics()
        const [h, w] = frame.depthSize
        const overlay = depthPreviewFromF32(frame.depth, w, h, view.width, view.height)
        pendingOverlay = { depthCanvas: overlay, boxes: frame.detections || [], frameId: frame.id }
        drawOverlay()
        lastFrameResult = frame
        $('saveVDSX').disabled = false
        log('Image processed.', 'ok')
    } catch (err) {
        log('Process error: ' + err.message, 'error')
    } finally {
        hideLoading()
    }
}

async function saveVDSXSingle(result, retentionMode) {
    const intr = currentIntrinsics || (await ensureIntrinsics())
    const { depth, depthSize, detections } = result
    const [h, w] = depthSize
    const metersPerUnit = Number($('scaleMeters').value) || 1.0
    const zip = new JSZip()
    const now = new Date().toISOString()
    const manifest = {
        vdsx_version: '1.5',
        created_utc: now,
        kind: 'image',
        dimensions: { width: view.width, height: view.height },
        retention_mode: retentionMode,
        layers: {
            rgb: retentionMode === 'FULL',
            intrinsics: true,
            extrinsics: true,
            sensors: { gps: false, imu: false },
            semantics: { yolo: detections?.length ? 1 : 0, seg: 0 },
            depth: {
                format: 'float32-bin',
                frames: 1,
                size: [h, w],
                channels: 1,
                keyframes: [0],
                delta: 'none',
                meters_per_unit: metersPerUnit
            }
        }
    }
    __applyWatermark(manifest)
    zip.file('manifest.json', JSON.stringify(manifest, null, 2))
    zip.file(
        'calib/intrinsics.json',
        JSON.stringify(
            {
                width: view.width,
                height: view.height,
                fx: intr.K.fx,
                fy: intr.K.fy,
                cx: intr.K.cx,
                cy: intr.K.cy,
                skew: intr.K.skew || 0,
                model: intr.model,
                source: intr.source,
                confidence: intr.confidence,
                qc: intr.qc
            },
            null,
            2
        )
    )
    zip.file(
        'calib/extrinsics.json',
        JSON.stringify(
            {
                R: [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ],
                t: [0, 0, 0],
                note: 'Single-view; scale via meters_per_unit.'
            },
            null,
            2
        )
    )
    zip.file('depth/depth_0001.bin', new Blob([depth.buffer], { type: 'application/octet-stream' }))
    zip.file(
        'depth/depth.meta.json',
        JSON.stringify(
            {
                dtype: 'float32',
                shape: [h, w],
                count: 1,
                stride: 1,
                keyframes: [0],
                meters_per_unit: metersPerUnit
            },
            null,
            2
        )
    )
    if (detections && detections.length) {
        zip.file('semantics/yolo.json', JSON.stringify({ frames: [detections] }, null, 2))
    }
    if (retentionMode === 'FULL') {
        const poster = await new Promise((res) =>
            view.toBlob((b) => res(b), 'image/jpeg', 0.92)
        )
        if (poster) zip.file('rgb/poster.jpg', poster)
        const Kd = scaleIntrinsicsTo({
            K: intr.K,
            fromW: view.width,
            fromH: view.height,
            toW: w,
            toH: h
        })
        const rgbCanvas = document.createElement('canvas')
        rgbCanvas.width = w
        rgbCanvas.height = h
        const rgbCtx = rgbCanvas.getContext('2d')
        rgbCtx.drawImage(view, 0, 0, view.width, view.height, 0, 0, w, h)
        const { xyz, colorsU8 } = backprojectToPointCloudFloat32({
            depthF32: depth,
            H: h,
            W: w,
            Kd,
            rgbCtx
        })
        zip.file('pointcloud/points_0001.bin', new Blob([xyz.buffer], { type: 'application/octet-stream' }))
        if (colorsU8) {
            zip.file('pointcloud/colors_0001.bin', new Blob([colorsU8.buffer], { type: 'application/octet-stream' }))
        }
        zip.file(
            'pointcloud/points.meta.json',
            JSON.stringify(
                {
                    dtype: 'float32',
                    layout: 'xyz',
                    size: [h, w],
                    count: 1,
                    colors: colorsU8 ? 'uint8x3' : null
                },
                null,
                2
            )
        )
    }
    zip.file(
        'meta/provenance.json',
        JSON.stringify(
            __watermarkProvenance({
                tool: 'Image→VDSX (keyframe pipeline)',
                ts: now,
                ep: $('s_ep').textContent.replace('EP: ', ''),
                notes: 'Generated asynchronously with keyframe propagation.'
            }),
            null,
            2
        )
    )
    const blob = await zip.generateAsync({
        type: 'blob',
        compression: 'DEFLATE',
        compressionOptions: { level: 5 }
    })
    const name = `vdsx_frame_${view.width}x${view.height}_${Date.now()}.vdsx`
    const a = document.createElement('a')
    a.download = name
    a.href = URL.createObjectURL(blob)
    document.body.appendChild(a)
    a.click()
    setTimeout(() => {
        URL.revokeObjectURL(a.href)
        document.body.removeChild(a)
    }, 600)
    log(`Saved ${name}`, 'ok')
}

$('saveVDSX').onclick = async (e) => {
    e.preventDefault()
    if (!lastFrameResult) {
        log('No frame available.', 'warn')
        return
    }
    showLoading()
    try {
        await saveVDSXSingle(lastFrameResult, $('retentionMode').value)
    } catch (err) {
        log('Save error: ' + err.message, 'error')
    } finally {
        hideLoading()
    }
}

$('startRec').onclick = (e) => {
    e.preventDefault()
    if (recording) return
    recording = true
    recZip = new JSZip()
    recFrames = 0
    recDepthFrames = []
    recDepthKeyframes = []
    recDetectionKeyframes = []
    recYOLO = []
    recRGB = []
    recIntrByFrame = []
    recPointXYZ = []
    recPointRGB = []
    recConfig = {
        retention: $('retentionMode').value,
        metersPerUnit: Number($('scaleMeters').value) || 1.0
    }
    recRawCount = 0
    chip($('s_rec'), 'REC: 0 frames', true)
    $('stopRec').disabled = false
    log('Recording started.', 'ok')
}

$('stopRec').onclick = async (e) => {
    e.preventDefault()
    if (!recording) return
    recording = false
    chip($('s_rec'), 'REC: idle')
    $('stopRec').disabled = true
    showLoading()
    try {
        await finalizeRecordingVDSX()
    } catch (err) {
        log('Record save error: ' + err.message, 'error')
    } finally {
        hideLoading()
    }
}

async function finalizeRecordingVDSX() {
    const frameCount = recDepthFrames.length
    if (!frameCount) {
        log('No frames captured.', 'warn')
        return
    }
    const intr = currentIntrinsics || (await ensureIntrinsics())
    const metersPerUnit = recConfig.metersPerUnit
    const now = new Date().toISOString()
    const manifest = {
        vdsx_version: '1.5',
        created_utc: now,
        kind: 'video',
        dimensions: { width: view.width, height: view.height },
        retention_mode: recConfig.retention,
        layers: {
            rgb: recConfig.retention === 'FULL',
            intrinsics: true,
            extrinsics: true,
            sensors: { gps: false, imu: false },
            semantics: { yolo: 1, seg: 0 },
            depth: {
                format: 'float32-bin',
                frames: frameCount,
                size: recDepthFrames[0].size,
                channels: 1,
                keyframes: recDepthKeyframes,
                delta: 'flow+propagated',
                meters_per_unit: metersPerUnit
            }
        }
    }
    if (recDetectionKeyframes.length) {
        manifest.layers.semantics.yolo_keyframes = recDetectionKeyframes
    }
    __applyWatermark(manifest)
    recZip.file('manifest.json', JSON.stringify(manifest, null, 2))
    recZip.file(
        'calib/intrinsics.json',
        JSON.stringify(
            {
                width: view.width,
                height: view.height,
                fx: intr.K.fx,
                fy: intr.K.fy,
                cx: intr.K.cx,
                cy: intr.K.cy,
                skew: intr.K.skew || 0,
                model: intr.model
            },
            null,
            2
        )
    )
    recZip.file(
        'calib/extrinsics.json',
        JSON.stringify(
            {
                R: [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ],
                t: [0, 0, 0],
                note: 'Single-view stream; scale via meters_per_unit.'
            },
            null,
            2
        )
    )
    recZip.file(
        'calib/frames/intrinsics.index.json',
        JSON.stringify({ frames: recIntrByFrame }, null, 2)
    )
    const [h, w] = recDepthFrames[0].size
    recDepthFrames.forEach((frame, i) => {
        const idx = String(i + 1).padStart(4, '0')
        recZip.file(
            `depth/depth_${idx}.bin`,
            new Blob([frame.data.buffer], { type: 'application/octet-stream' })
        )
    })
    recZip.file(
        'depth/depth.meta.json',
        JSON.stringify(
            {
                dtype: 'float32',
                shape: [h, w],
                count: frameCount,
                stride: 1,
                keyframes: recDepthKeyframes,
                meters_per_unit: metersPerUnit
            },
            null,
            2
        )
    )
    recZip.file(
        'semantics/yolo.json',
        JSON.stringify({ frames: recYOLO, keyframes: recDetectionKeyframes }, null, 2)
    )
    if (recConfig.retention === 'FULL') {
        recRGB.forEach((blob, i) => {
            const idx = String(i + 1).padStart(4, '0')
            recZip.file(`rgb/frame_${idx}.jpg`, blob)
        })
        recPointXYZ.forEach((xyz, i) => {
            const idx = String(i + 1).padStart(4, '0')
            recZip.file(
                `pointcloud/points_${idx}.bin`,
                new Blob([xyz.buffer], { type: 'application/octet-stream' })
            )
            if (i === 0) {
                recZip.file(
                    'pointcloud/points.bin',
                    new Blob([xyz.buffer], { type: 'application/octet-stream' })
                )
            }
            const rgb = recPointRGB[i]
            if (rgb) {
                recZip.file(
                    `pointcloud/colors_${idx}.bin`,
                    new Blob([rgb.buffer], { type: 'application/octet-stream' })
                )
            }
        })
        recZip.file(
            'pointcloud/points.meta.json',
            JSON.stringify(
                {
                    dtype: 'float32',
                    layout: 'xyz',
                    size: [h, w],
                    count: frameCount,
                    colors: recPointRGB[0] ? 'uint8x3' : null
                },
                null,
                2
            )
        )
    }
    recZip.file(
        'meta/provenance.json',
        JSON.stringify(
            __watermarkProvenance({
                tool: 'Video→VDSX (keyframe pipeline)',
                ts: now,
                ep: $('s_ep').textContent.replace('EP: ', ''),
                notes: 'Depth/detections decimated with optical-flow propagation.'
            }),
            null,
            2
        )
    )
    const blob = await recZip.generateAsync({
        type: 'blob',
        compression: 'DEFLATE',
        compressionOptions: { level: 5 }
    })
    const name = `vdsx_video_${view.width}x${view.height}_${frameCount}f_${Date.now()}.vdsx`
    const a = document.createElement('a')
    a.download = name
    a.href = URL.createObjectURL(blob)
    document.body.appendChild(a)
    a.click()
    setTimeout(() => {
        URL.revokeObjectURL(a.href)
        document.body.removeChild(a)
    }, 600)
    log(`Saved ${name}`, 'ok')
    recZip = null
    recDepthFrames = []
    recYOLO = []
    recRGB = []
    recIntrByFrame = []
    recPointXYZ = []
    recPointRGB = []
    recRawCount = 0
}

window.addEventListener('error', (e) => {
    console.error(e)
    log('App error: ' + (e.error?.message || e.message || 'Unknown'), 'error')
    hideLoading()
})
window.addEventListener('unhandledrejection', (e) => {
    console.error(e.reason)
    log('Async error: ' + (e.reason?.message || 'Unknown'), 'error')
    hideLoading()
    e.preventDefault()
})

document.querySelectorAll('input[name="src"]').forEach((r) => {
    r.addEventListener('change', () => {
        const v = r.value
        $('imageControls').style.display = v === 'image' ? '' : 'none'
        $('videoControls').style.display = v === 'video' ? '' : 'none'
        $('video').style.display = v === 'video' ? '' : 'none'
        $('startRec').disabled = v !== 'video' || !depthModelReady
        $('stopRec').disabled = true
    })
})

log('Ready. Load a depth model; then pick Image or start Camera.', 'info')
