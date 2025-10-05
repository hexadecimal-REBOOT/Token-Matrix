const BLOCK = 8
const SEARCH = 4

function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v))
}

function computeFlow(prev, curr, width, height) {
    const dx = new Float32Array(width * height)
    const dy = new Float32Array(width * height)
    const prevView = new Uint8ClampedArray(prev)
    const currView = new Uint8ClampedArray(curr)
    for (let by = 0; by < height; by += BLOCK) {
        for (let bx = 0; bx < width; bx += BLOCK) {
            let bestDx = 0
            let bestDy = 0
            let bestScore = Infinity
            for (let oy = -SEARCH; oy <= SEARCH; oy++) {
                for (let ox = -SEARCH; ox <= SEARCH; ox++) {
                    let score = 0
                    for (let iy = 0; iy < BLOCK; iy++) {
                        const sy = clamp(by + iy, 0, height - 1)
                        const ty = clamp(sy + oy, 0, height - 1)
                        for (let ix = 0; ix < BLOCK; ix++) {
                            const sx = clamp(bx + ix, 0, width - 1)
                            const tx = clamp(sx + ox, 0, width - 1)
                            const idxSrc = sy * width + sx
                            const idxTgt = ty * width + tx
                            const diff =
                                prevView[idxSrc] - currView[idxTgt]
                            score += Math.abs(diff)
                        }
                    }
                    if (score < bestScore) {
                        bestScore = score
                        bestDx = -ox
                        bestDy = -oy
                    }
                }
            }
            for (let iy = 0; iy < BLOCK; iy++) {
                const sy = clamp(by + iy, 0, height - 1)
                for (let ix = 0; ix < BLOCK; ix++) {
                    const sx = clamp(bx + ix, 0, width - 1)
                    const idx = sy * width + sx
                    dx[idx] = bestDx
                    dy[idx] = bestDy
                }
            }
        }
    }
    return { dx, dy }
}

self.onmessage = (e) => {
    const msg = e.data
    if (msg.type === 'configure') {
        return
    }
    if (msg.type === 'compute') {
        const { dx, dy } = computeFlow(
            msg.prev,
            msg.curr,
            msg.width,
            msg.height
        )
        self.postMessage(
            {
                type: 'flow',
                prevId: msg.prevId,
                currId: msg.currId,
                width: msg.width,
                height: msg.height,
                dx: dx.buffer,
                dy: dy.buffer
            },
            [dx.buffer, dy.buffer]
        )
    }
}
