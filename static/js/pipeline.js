const DEFAULT_DEPTH_STRIDE = 10
const DEFAULT_YOLO_STRIDE = 45
const MAX_QUEUE = 4
const YOLO_INPUT_SIZE = 640

const hasOffscreen = typeof OffscreenCanvas !== 'undefined'

function makeCanvas(width, height) {
    if (hasOffscreen) {
        const canvas = new OffscreenCanvas(width, height)
        return canvas
    }
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    return canvas
}

let WORKER_ID = 0
function workerId(prefix) {
    WORKER_ID += 1
    return `${prefix}_${WORKER_ID}`
}

function averageFlow(flow, width, height, box) {
    const x0 = Math.max(0, Math.floor(box.x))
    const y0 = Math.max(0, Math.floor(box.y))
    const x1 = Math.min(width - 1, Math.ceil(box.x + box.w))
    const y1 = Math.min(height - 1, Math.ceil(box.y + box.h))
    const dxField = flow.dx
    const dyField = flow.dy
    let sumX = 0
    let sumY = 0
    let count = 0
    for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
            const idx = y * width + x
            sumX += dxField[idx]
            sumY += dyField[idx]
            count++
        }
    }
    if (!count) return { dx: 0, dy: 0 }
    return { dx: sumX / count, dy: sumY / count }
}

function propagateDepthForward(prevDepth, flow, width, height) {
    const out = new Float32Array(prevDepth.length)
    const dx = flow.dx
    const dy = flow.dy
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x
            const sx = x - dx[idx]
            const sy = y - dy[idx]
            const ix = Math.max(0, Math.min(width - 1, sx))
            const iy = Math.max(0, Math.min(height - 1, sy))
            const x0 = Math.floor(ix)
            const y0 = Math.floor(iy)
            const x1 = Math.min(width - 1, x0 + 1)
            const y1 = Math.min(height - 1, y0 + 1)
            const wx = ix - x0
            const wy = iy - y0
            const idx00 = y0 * width + x0
            const idx10 = y0 * width + x1
            const idx01 = y1 * width + x0
            const idx11 = y1 * width + x1
            const z00 = prevDepth[idx00]
            const z10 = prevDepth[idx10]
            const z01 = prevDepth[idx01]
            const z11 = prevDepth[idx11]
            const z0 = z00 + wx * (z10 - z00)
            const z1 = z01 + wx * (z11 - z01)
            out[idx] = z0 + wy * (z1 - z0)
        }
    }
    return out
}

function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val))
}

let TRACK_ID = 0
function nextTrackId() {
    TRACK_ID += 1
    return TRACK_ID
}

function iou(a, b) {
    const ax1 = a.x
    const ay1 = a.y
    const ax2 = a.x + a.w
    const ay2 = a.y + a.h
    const bx1 = b.x
    const by1 = b.y
    const bx2 = b.x + b.w
    const by2 = b.y + b.h
    const ix1 = Math.max(ax1, bx1)
    const iy1 = Math.max(ay1, by1)
    const ix2 = Math.min(ax2, bx2)
    const iy2 = Math.min(ay2, by2)
    const iw = Math.max(0, ix2 - ix1)
    const ih = Math.max(0, iy2 - iy1)
    const inter = iw * ih
    if (!inter) return 0
    const union = a.w * a.h + b.w * b.h - inter
    return union > 0 ? inter / union : 0
}

export class CapturePipeline {
    constructor({
        log,
        onDepth,
        onDetections,
        onFrameReady,
        onFlow,
        requestIntrinsics
    }) {
        this.log = log
        this.onDepth = onDepth
        this.onDetections = onDetections
        this.onFrameReady = onFrameReady
        this.onFlow = onFlow
        this.requestIntrinsics = requestIntrinsics

        this.depthWorker = null
        this.detWorker = null
        this.flowWorker = null

        this.depthStride = DEFAULT_DEPTH_STRIDE
        this.detStride = DEFAULT_YOLO_STRIDE
        this.depthInputSize = 320

        this._flowCanvas = null
        this._flowCtx = null
        this._detCanvas = null
        this._detCtx = null

        this.frames = new Map()
        this.frameQueue = []
        this.running = false
        this.frameId = 0
        this.lastDepthFrame = null
        this.lastDetFrame = null
        this.pendingDepth = new Set()
        this.pendingDet = new Set()
        this.tracks = []
        this.metersPerUnit = 1.0

        this.flowWidth = 0
        this.flowHeight = 0

        this.stats = {
            depthHz: 0,
            detHz: 0,
            frameHz: 0
        }

        this._depthTimes = []
        this._detTimes = []
        this._frameTimes = []
    }

    async init() {
        if (!this.flowWorker) {
            const flowUrl = new URL('./workers/flowWorker.js', import.meta.url)
            this.flowWorker = new Worker(flowUrl, {
                name: workerId('flow'),
                type: 'classic'
            })
            this.flowWorker.onmessage = (e) => this._handleFlow(e.data)
        }
        if (!this.depthWorker) {
            const depthUrl = new URL(
                './workers/depthWorker.js',
                import.meta.url
            )
            this.depthWorker = new Worker(depthUrl, {
                name: workerId('depth'),
                type: 'classic'
            })
            this.depthWorker.onmessage = (e) => this._handleDepth(e.data)
        }
        if (!this.detWorker) {
            const detUrl = new URL(
                './workers/detectWorker.js',
                import.meta.url
            )
            this.detWorker = new Worker(detUrl, {
                name: workerId('det'),
                type: 'classic'
            })
            this.detWorker.onmessage = (e) => this._handleDetections(e.data)
        }
    }

    dispose() {
        this.flowWorker?.terminate()
        this.depthWorker?.terminate()
        this.detWorker?.terminate()
        this.flowWorker = null
        this.depthWorker = null
        this.detWorker = null
    }

    setMetersPerUnit(v) {
        this.metersPerUnit = v
    }

    setDepthStride(v) {
        this.depthStride = clamp(Math.round(v), 1, 120)
    }

    setDetStride(v) {
        this.detStride = clamp(Math.round(v), 1, 240)
    }

    setDepthInputSize(v) {
        this.depthInputSize = clamp(Math.round(v), 128, 512)
        this.flowWidth = this.depthInputSize
        this.flowHeight = this.depthInputSize
        if (this.flowWorker) {
            this.flowWorker.postMessage({
                type: 'configure',
                width: this.flowWidth,
                height: this.flowHeight
            })
        }
    }

    async loadDepthModel(buffer, { executionProvider = 'wasm' } = {}) {
        await this.init()
        this.depthWorker.postMessage(
            {
                type: 'init',
                buffer,
                executionProvider
            },
            [buffer]
        )
    }

    async loadDetModel(buffer, { executionProvider = 'wasm' } = {}) {
        await this.init()
        this.detWorker.postMessage(
            {
                type: 'init',
                buffer,
                executionProvider
            },
            [buffer]
        )
    }

    async enqueueFrame(bitmap, width, height, timestamp) {
        if (!this.running) return
        if (this.frameQueue.length >= MAX_QUEUE) {
            bitmap.close()
            return
        }
        const id = this.frameId++
        const frame = {
            id,
            timestamp,
            width,
            height,
            bitmap,
            depth: null,
            depthSource: null,
            detections: [],
            detSource: null,
            flowFromPrev: null
        }
        this.frames.set(id, frame)
        this.frameQueue.push(frame)
        this._ensureFlowFrame(frame)
        this._updateFrameHz(timestamp)

        if (id === 0) {
            this._runDepth(frame, true)
            this._runDetection(frame, true)
        }

        const prev = this.frames.get(id - 1)
        if (prev) {
            this._requestFlow(prev, frame)
        }

        const runDepth = id % this.depthStride === 0
        const runDet = id % this.detStride === 0

        if (runDet) {
            this._runDetection(frame, true)
        }
        if (runDepth) {
            this._runDepth(frame, true)
        }
        if (!runDepth && !runDet && frame.bitmap) {
            frame.bitmap.close()
            frame.bitmap = null
        }
    }

    start() {
        this.running = true
        this.frameQueue.length = 0
        this.frames.clear()
        this.frameId = 0
        this.lastDepthFrame = null
        this.lastDetFrame = null
        this.tracks = []
    }

    stop() {
        this.running = false
        for (const frame of this.frameQueue) {
            frame.bitmap?.close?.()
        }
        this.frameQueue.length = 0
    }

    async processImage(bitmap, width, height) {
        await this.init()
        return new Promise((resolve) => {
            const id = this.frameId++
            const frame = {
                id,
                timestamp: performance.now(),
                width,
                height,
                bitmap,
                depth: null,
                depthSource: null,
                detections: [],
                detSource: null,
                flowFromPrev: null,
                resolve
            }
            this.frames.set(id, frame)
            this._ensureFlowFrame(frame)
            this._runDetection(frame, true)
            this._runDepth(frame, true)
        })
    }

    _ensureFlowFrame(frame) {
        const target = this.flowWidth || this.depthInputSize
        if (frame.flowGray && frame.flowGray.width === target) return frame.flowGray
        if (!frame.bitmap) return frame.flowGray || null
        if (!this._flowCanvas || this._flowCanvas.width !== target) {
            this._flowCanvas = makeCanvas(target, target)
            this._flowCtx = this._flowCanvas.getContext('2d', {
                willReadFrequently: true
            })
        }
        const ctx = this._flowCtx
        ctx.clearRect(0, 0, target, target)
        ctx.drawImage(
            frame.bitmap,
            0,
            0,
            frame.width,
            frame.height,
            0,
            0,
            target,
            target
        )
        const img = ctx.getImageData(0, 0, target, target).data
        const gray = new Uint8ClampedArray(target * target)
        for (let i = 0, j = 0; i < img.length; i += 4, j++) {
            gray[j] = Math.round(
                0.299 * img[i] + 0.587 * img[i + 1] + 0.114 * img[i + 2]
            )
        }
        frame.flowGray = { data: gray, width: target, height: target }
        return frame.flowGray
    }

    _prepareDetectionInput(frame) {
        if (!frame.bitmap) return null
        const size = YOLO_INPUT_SIZE
        if (!this._detCanvas || this._detCanvas.width !== size) {
            this._detCanvas = makeCanvas(size, size)
            this._detCtx = this._detCanvas.getContext('2d', {
                willReadFrequently: true
            })
        }
        const ctx = this._detCtx
        ctx.fillStyle = '#000'
        ctx.fillRect(0, 0, size, size)
        const scale = Math.min(size / frame.width, size / frame.height)
        const nw = Math.round(frame.width * scale)
        const nh = Math.round(frame.height * scale)
        const offx = ((size - nw) / 2) | 0
        const offy = ((size - nh) / 2) | 0
        ctx.drawImage(
            frame.bitmap,
            0,
            0,
            frame.width,
            frame.height,
            offx,
            offy,
            nw,
            nh
        )
        const rgba = ctx.getImageData(0, 0, size, size).data
        const area = size * size
        const tensor = new Float32Array(3 * area)
        for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
            tensor[p] = rgba[i] / 255
            tensor[p + area] = rgba[i + 1] / 255
            tensor[p + 2 * area] = rgba[i + 2] / 255
        }
        return {
            tensor,
            pad: { offx, offy, scale, size }
        }
    }

    _requestFlow(prev, curr) {
        if (!this.flowWorker) return
        const target = this.flowWidth || this.depthInputSize
        const prevGray = this._ensureFlowFrame(prev)
        const currGray = this._ensureFlowFrame(curr)
        if (!prevGray || !currGray) return
        const prevBuffer = prevGray.data.buffer.slice(0)
        const currBuffer = currGray.data.buffer.slice(0)
        this.flowWorker.postMessage(
            {
                type: 'compute',
                prevId: prev.id,
                currId: curr.id,
                width: target,
                height: target,
                prev: prevBuffer,
                curr: currBuffer
            },
            [prevBuffer, currBuffer]
        )
    }

    _runDepth(frame, markKeyframe) {
        if (!this.depthWorker || this.pendingDepth.has(frame.id)) return
        this.pendingDepth.add(frame.id)
        if (!frame.bitmap) {
            this.pendingDepth.delete(frame.id)
            this.log?.('Depth bitmap missing for frame ' + frame.id)
            return
        }
        this.depthWorker.postMessage(
            {
                type: 'run',
                frameId: frame.id,
                targetSize: this.depthInputSize,
                metersPerUnit: this.metersPerUnit,
                bitmap: frame.bitmap
            },
            [frame.bitmap]
        )
        frame.bitmap = null
        if (markKeyframe) {
            frame.depthKeyframe = true
        }
    }

    _runDetection(frame, markKeyframe) {
        if (!this.detWorker || this.pendingDet.has(frame.id)) return
        this.pendingDet.add(frame.id)
        const prep = this._prepareDetectionInput(frame)
        if (!prep) {
            this.pendingDet.delete(frame.id)
            return
        }
        this.detWorker.postMessage(
            {
                type: 'run',
                frameId: frame.id,
                tensor: prep.tensor.buffer,
                pad: prep.pad,
                origWidth: frame.width,
                origHeight: frame.height
            },
            [prep.tensor.buffer]
        )
        if (markKeyframe) {
            frame.detKeyframe = true
        }
    }

    _handleFlow(msg) {
        if (msg.type !== 'flow') return
        const prev = this.frames.get(msg.prevId)
        const curr = this.frames.get(msg.currId)
        if (!prev || !curr) return
        curr.flowFromPrev = {
            width: msg.width,
            height: msg.height,
            dx: new Float32Array(msg.dx),
            dy: new Float32Array(msg.dy)
        }
        this.onFlow?.(curr.id, curr.flowFromPrev)
        if (prev.depth) {
            this._propagateDepth(prev, curr)
        }
        if (prev.detections?.length) {
            this._propagateDetections(prev, curr)
        }
        this._maybeFinalize(curr)
    }

    _handleDepth(msg) {
        if (msg.type === 'error') {
            this.log?.('Depth worker error: ' + msg.error)
            return
        }
        if (msg.type === 'ready') {
            this.log?.('Depth worker ready.')
            return
        }
        if (msg.type !== 'depth') return
        this.pendingDepth.delete(msg.frameId)
        const frame = this.frames.get(msg.frameId)
        if (!frame) return
        frame.depth = new Float32Array(msg.data)
        frame.depthSize = msg.size
        frame.depthSource = msg.source || 'keyframe'
        this.onDepth?.(frame)
        this._depthTimes.push(performance.now())
        this._updateHz('depthHz', this._depthTimes)
        if (frame.id > 0) {
            const next = this.frames.get(frame.id + 1)
            if (next && next.flowFromPrev) {
                this._propagateDepth(frame, next)
            }
        }
        this._maybeFinalize(frame)
    }

    _handleDetections(msg) {
        if (msg.type === 'error') {
            this.log?.('Detection worker error: ' + msg.error)
            return
        }
        if (msg.type === 'ready') {
            this.log?.('Detection worker ready.')
            return
        }
        if (msg.type !== 'detections') return
        this.pendingDet.delete(msg.frameId)
        const frame = this.frames.get(msg.frameId)
        if (!frame) return
        const dets = msg.detections
        this._updateTracks(frame, dets, true)
        frame.detections = this.tracks.map((t) => ({
            id: t.id,
            x: t.box.x,
            y: t.box.y,
            w: t.box.w,
            h: t.box.h,
            conf: t.conf,
            cls: t.cls
        }))
        frame.detSource = msg.source || 'keyframe'
        this.onDetections?.(frame)
        this._detTimes.push(performance.now())
        this._updateHz('detHz', this._detTimes)
        if (frame.id > 0) {
            const next = this.frames.get(frame.id + 1)
            if (next && next.flowFromPrev) {
                this._propagateDetections(frame, next)
            }
        }
        this._maybeFinalize(frame)
    }

    _propagateDepth(prev, curr) {
        if (!curr.flowFromPrev || !prev.depth) return
        const flow = curr.flowFromPrev
        const depth = propagateDepthForward(
            prev.depth,
            flow,
            flow.width,
            flow.height
        )
        curr.depth = depth
        curr.depthSize = [flow.height, flow.width]
        curr.depthSource = 'propagated'
        this.onDepth?.(curr)
    }

    _updateTracks(frame, detections, isKeyframe) {
        const updatedIds = new Set()
        const assigned = new Set()
        for (const det of detections) {
            let best = null
            let bestScore = 0
            for (const track of this.tracks) {
                if (assigned.has(track.id)) continue
                const score = iou(track.box, det)
                if (score > bestScore) {
                    bestScore = score
                    best = track
                }
            }
            if (best && bestScore > 0.2) {
                const flow = frame.flowFromPrev
                if (flow) {
                    const delta = averageFlow(
                        flow,
                        flow.width,
                        flow.height,
                        det
                    )
                    best.velocity = { dx: delta.dx, dy: delta.dy }
                }
                best.box = { ...det }
                best.conf = det.conf
                best.cls = det.cls
                best.lastFrame = frame.id
                assigned.add(best.id)
                updatedIds.add(best.id)
            } else {
                const id = nextTrackId()
                this.tracks.push({
                    id,
                    box: { ...det },
                    conf: det.conf,
                    cls: det.cls,
                    velocity: { dx: 0, dy: 0 },
                    lastFrame: frame.id
                })
                updatedIds.add(id)
            }
        }
        const retained = []
        for (const track of this.tracks) {
            if (frame.id - track.lastFrame > this.detStride * 2) {
                continue
            }
            if (!updatedIds.has(track.id) && frame.flowFromPrev) {
                const delta = averageFlow(
                    frame.flowFromPrev,
                    frame.flowFromPrev.width,
                    frame.flowFromPrev.height,
                    track.box
                )
                track.box = {
                    ...track.box,
                    x: track.box.x + delta.dx,
                    y: track.box.y + delta.dy
                }
                track.velocity = delta
                track.conf *= 0.95
                track.lastFrame = frame.id
            }
            retained.push(track)
        }
        this.tracks = retained
    }

    _propagateDetections(prev, curr) {
        if (!curr.flowFromPrev) return
        const nextTracks = []
        for (const track of this.tracks) {
            if (curr.id - track.lastFrame > this.detStride * 2) continue
            const delta = averageFlow(
                curr.flowFromPrev,
                curr.flowFromPrev.width,
                curr.flowFromPrev.height,
                track.box
            )
            const box = {
                ...track.box,
                x: track.box.x + delta.dx,
                y: track.box.y + delta.dy
            }
            nextTracks.push({
                ...track,
                box,
                velocity: delta,
                lastFrame: curr.id,
                conf: track.conf * 0.97
            })
        }
        this.tracks = nextTracks
        curr.detections = this.tracks.map((t) => ({
            id: t.id,
            x: t.box.x,
            y: t.box.y,
            w: t.box.w,
            h: t.box.h,
            conf: t.conf,
            cls: t.cls
        }))
        curr.detSource = 'propagated'
        this.onDetections?.(curr)
    }

    _maybeFinalize(frame) {
        if (frame.depth && frame.detections) {
            this._finalizeFrame(frame)
        }
    }

    _finalizeFrame(frame) {
        this.onFrameReady?.(frame)
        if (frame.resolve) {
            frame.resolve(frame)
        }
        const threshold = frame.id - MAX_QUEUE
        for (const [id, st] of this.frames) {
            if (id < threshold) {
                st.bitmap?.close?.()
                this.frames.delete(id)
            }
        }
    }

    _updateFrameHz(ts) {
        this._frameTimes.push(ts)
        this._updateHz('frameHz', this._frameTimes)
    }

    _updateHz(field, arr) {
        const now = performance.now()
        while (arr.length && now - arr[0] > 2000) arr.shift()
        if (arr.length > 1) {
            const duration = arr[arr.length - 1] - arr[0]
            if (duration > 0) {
                this.stats[field] = ((arr.length - 1) / duration) * 1000
            }
        }
    }
}
