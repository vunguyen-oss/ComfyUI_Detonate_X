/**
 * ComfyUI_Detonate - RotoBezier Interactive Widget
 * 
 * Professional Rotoscoping UI:
 * - Aspect Ratio Support: Centers and scales drawing area to match input image
 * - Pen Tool: Left-Click (Add/Move), Right-Click (Remove)
 * - Clear Button: Fully resets state and synchronizes
 */

import { app } from "../../scripts/app.js";

class RotoBezierCanvas {
    constructor(node, splineWidget) {
        this.node = node;
        this.splineWidget = splineWidget;
        this.splines = this.parseData(splineWidget.value);

        this.activeSpline = null;
        this.selectedPoint = null;
        this.selectedHandle = null;
        this.hovered = null;
        this.dragging = false;

        // Container
        this.container = document.createElement("div");
        this.container.style.width = "100%";
        this.container.style.height = "100%";
        this.container.style.backgroundColor = "#080808";
        this.container.style.position = "relative";
        this.container.style.overflow = "hidden";
        this.container.style.border = "1px solid #333";

        // Canvas Layer
        this.canvas = document.createElement("canvas");
        this.canvas.style.width = "100%";
        this.canvas.style.height = "100%";
        this.canvas.style.cursor = "crosshair";
        this.canvas.style.display = "block";
        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");

        // UI Layer
        this.overlay = document.createElement("div");
        this.overlay.style.position = "absolute";
        this.overlay.style.top = "5px";
        this.overlay.style.right = "5px";
        this.overlay.style.display = "flex";
        this.overlay.style.gap = "4px";
        this.container.appendChild(this.overlay);

        this.addOverlayButton("Clear", () => {
            if (confirm("Clear all splines?")) {
                this.splines = [];
                this.activeSpline = null;
                this.selectedPoint = null;
                this.selectedHandle = null;
                this.updateData();
                this.render();
            }
        });

        // Initialize
        this.resizeObserver = new ResizeObserver(() => this.onResize());
        this.resizeObserver.observe(this.container);

        // Bind Events
        this.canvas.addEventListener("mousedown", this.onMouseDown.bind(this));
        this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

        this._mouseMoveHandler = this.onMouseMove.bind(this);
        this._mouseUpHandler = this.onMouseUp.bind(this);
        window.addEventListener("mousemove", this._mouseMoveHandler);
        window.addEventListener("mouseup", this._mouseUpHandler);

        // Viewport stats
        this.viewport = { x: 0, y: 0, w: 0, h: 0 };

        // Frame loop
        this._boundRender = this.render.bind(this);
        requestAnimationFrame(this._boundRender);
    }

    addOverlayButton(text, cb) {
        const btn = document.createElement("button");
        btn.innerText = text;
        btn.style.padding = "4px 10px";
        btn.style.fontSize = "11px";
        btn.style.backgroundColor = "#222";
        btn.style.color = "#eee";
        btn.style.border = "1px solid #444";
        btn.style.borderRadius = "3px";
        btn.style.cursor = "pointer";
        btn.style.zIndex = "10";
        btn.onmousedown = (e) => e.stopPropagation();
        btn.onclick = (e) => { e.stopPropagation(); cb(); };
        this.overlay.appendChild(btn);
    }

    parseData(val) {
        try {
            const d = JSON.parse(val || '{"splines":[]}');
            return d.splines || [];
        } catch (e) { return []; }
    }

    updateData() {
        this.splineWidget.value = JSON.stringify({ splines: this.splines });
        this.node.setDirtyCanvas(true, true);
    }

    onResize() {
        if (!this.container.clientWidth) return;
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = this.container.clientWidth * dpr;
        this.canvas.height = this.container.clientHeight * dpr;
        this.render();
    }

    // Coordinate mapping with Aspect Ratio logic
    updateViewport(img) {
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        if (!img) {
            this.viewport = { x: 0, y: 0, w: cw, h: ch };
            return;
        }

        const imgRatio = img.width / img.height;
        const canvasRatio = cw / ch;

        let vw, vh;
        if (imgRatio > canvasRatio) {
            vw = cw;
            vh = cw / imgRatio;
        } else {
            vh = ch;
            vw = ch * imgRatio;
        }

        this.viewport = {
            x: (cw - vw) / 2,
            y: (ch - vh) / 2,
            w: vw,
            h: vh
        };
    }

    toCanvas(nx, ny) {
        return {
            x: this.viewport.x + nx * this.viewport.w,
            y: this.viewport.y + ny * this.viewport.h
        };
    }

    toNorm(clientX, clientY) {
        const rect = this.canvas.getBoundingClientRect();

        // Map client coords → canvas CSS coords → canvas pixel coords
        const cssX = clientX - rect.left;
        const cssY = clientY - rect.top;

        // Scale from CSS pixels to canvas pixels
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const ax = cssX * scaleX;
        const ay = cssY * scaleY;

        return {
            x: Math.max(0, Math.min(1, (ax - this.viewport.x) / this.viewport.w)),
            y: Math.max(0, Math.min(1, (ay - this.viewport.y) / this.viewport.h))
        };
    }

    findHit(px, py) {
        const threshold = 15 * (window.devicePixelRatio || 1);
        for (let sIdx = 0; sIdx < this.splines.length; sIdx++) {
            const s = this.splines[sIdx];
            for (let pIdx = 0; pIdx < s.points.length; pIdx++) {
                const p = s.points[pIdx];
                const pos = this.toCanvas(p.x, p.y);
                if (Math.hypot(pos.x - px, pos.y - py) < threshold) return { sIdx, pIdx, type: 'point' };

                const hIn = this.toCanvas(p.x + p.handleIn.x, p.y + p.handleIn.y);
                if (Math.hypot(hIn.x - px, hIn.y - py) < threshold) return { sIdx, pIdx, type: 'in' };

                const hOut = this.toCanvas(p.x + p.handleOut.x, p.y + p.handleOut.y);
                if (Math.hypot(hOut.x - px, hOut.y - py) < threshold) return { sIdx, pIdx, type: 'out' };
            }
        }
        return null;
    }

    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const px = (e.clientX - rect.left) * scaleX;
        const py = (e.clientY - rect.top) * scaleY;
        const norm = this.toNorm(e.clientX, e.clientY);

        if (e.button === 0) { // Left Click
            const hit = this.findHit(px, py);

            // Auto close check
            if (this.activeSpline !== null && hit && hit.sIdx === this.activeSpline && hit.pIdx === 0 && this.splines[this.activeSpline].points.length > 2) {
                this.splines[this.activeSpline].closed = true;
                this.activeSpline = null;
                this.updateData();
                return;
            }

            if (hit) {
                this.selectedPoint = hit.type === 'point' ? hit : null;
                this.selectedHandle = hit.type !== 'point' ? hit : null;
                this.dragging = true;
            } else {
                let targetSpline;
                if (this.activeSpline != null) {
                    targetSpline = this.splines[this.activeSpline];
                } else {
                    targetSpline = { points: [], closed: false, operation: 'add' };
                    this.splines.push(targetSpline);
                    this.activeSpline = this.splines.length - 1;
                }

                const newPoint = { x: norm.x, y: norm.y, handleIn: { x: 0, y: 0 }, handleOut: { x: 0, y: 0 }, smooth: true };
                targetSpline.points.push(newPoint);
                this.selectedPoint = { sIdx: this.splines.indexOf(targetSpline), pIdx: targetSpline.points.length - 1 };
                this.dragging = true;
                this.isNewPoint = true;
                this.updateData();
            }
        } else if (e.button === 2) { // Right Click
            const hit = this.findHit(px, py);
            if (hit && hit.type === 'point') {
                const s = this.splines[hit.sIdx];
                s.points.splice(hit.pIdx, 1);
                if (s.points.length === 0) {
                    this.splines.splice(hit.sIdx, 1);
                    if (this.activeSpline === hit.sIdx) this.activeSpline = null;
                }
                this.updateData();
            }
        }
        this.render();
    }


    onMouseMove(e) {
        const norm = this.toNorm(e.clientX, e.clientY);

        if (this.dragging) {
            if (this.isNewPoint && this.selectedPoint) {
                const s = this.splines[this.selectedPoint.sIdx];
                const p = s.points[this.selectedPoint.pIdx];
                p.handleOut.x = norm.x - p.x;
                p.handleOut.y = norm.y - p.y;
                p.handleIn.x = -p.handleOut.x;
                p.handleIn.y = -p.handleOut.y;
            } else if (this.selectedPoint) {
                const s = this.splines[this.selectedPoint.sIdx];
                const p = s.points[this.selectedPoint.pIdx];
                p.x = norm.x;
                p.y = norm.y;
            } else if (this.selectedHandle) {
                const s = this.splines[this.selectedHandle.sIdx];
                const p = s.points[this.selectedHandle.pIdx];
                if (this.selectedHandle.type === 'out') {
                    p.handleOut.x = norm.x - p.x;
                    p.handleOut.y = norm.y - p.y;
                    if (p.smooth) { p.handleIn.x = -p.handleOut.x; p.handleIn.y = -p.handleOut.y; }
                } else {
                    p.handleIn.x = norm.x - p.x;
                    p.handleIn.y = norm.y - p.y;
                    if (p.smooth) { p.handleOut.x = -p.handleIn.x; p.handleOut.y = -p.handleIn.y; }
                }
            }
            this.updateData();
            this.render();
        } else {
            const rect = this.canvas.getBoundingClientRect();
            const scaleX = this.canvas.width / rect.width;
            const scaleY = this.canvas.height / rect.height;
            const hit = this.findHit(
                (e.clientX - rect.left) * scaleX,
                (e.clientY - rect.top) * scaleY
            );
            if (hit != this.hovered) {
                this.hovered = hit;
                this.render();
            }
        }
    }

    onMouseUp() {
        this.dragging = false;
        this.isNewPoint = false;
    }

    findBgImage() {
        if (!this.node.inputs || !this.node.inputs[0]) return null;
        const linkId = this.node.inputs[0].link;
        if (linkId == null || !app.graph) return null;
        const link = app.graph.links[linkId];
        if (!link) return null;
        const originNode = app.graph.getNodeById(link.origin_id);
        if (!originNode) return null;
        return (originNode.imgs && originNode.imgs[0]) || originNode.image;
    }

    render() {
        if (!this.ctx || this.canvas.width === 0) return;
        const dpr = window.devicePixelRatio || 1;
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        ctx.clearRect(0, 0, w, h);

        const bgImg = this.findBgImage();
        this.updateViewport(bgImg);

        if (bgImg) {
            ctx.filter = "brightness(0.6) contrast(1.1)";
            ctx.globalAlpha = 0.5;
            ctx.drawImage(bgImg, this.viewport.x, this.viewport.y, this.viewport.w, this.viewport.h);
            ctx.globalAlpha = 1.0;
            ctx.filter = "none";
        } else {
            ctx.fillStyle = "#111";
            ctx.fillRect(0, 0, w, h);
            ctx.fillStyle = "#444";
            ctx.font = `${Math.round(14 * dpr)}px sans-serif`;
            ctx.textAlign = "center";
            ctx.fillText("Connect 'image' to see background ratio", w / 2, h / 2);
        }

        // Help draw viewport boundary
        ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(this.viewport.x, this.viewport.y, this.viewport.w, this.viewport.h);
        ctx.setLineDash([]);

        this.splines.forEach((s, sIdx) => {
            const points = s.points;
            if (points.length < 1) return;

            const isSelected = (this.selectedPoint?.sIdx === sIdx || this.selectedHandle?.sIdx === sIdx || this.activeSpline === sIdx);

            ctx.beginPath();
            ctx.strokeStyle = isSelected ? '#00ccff' : 'rgba(255,255,255,0.7)';
            ctx.lineWidth = 0.7 * dpr; // spline thickness
            ctx.lineJoin = "round";

            const start = this.toCanvas(points[0].x, points[0].y);
            ctx.moveTo(start.x, start.y);

            for (let i = 0; i < points.length - 1; i++) {
                const p1 = points[i];
                const p2 = points[i + 1];
                const c1 = this.toCanvas(p1.x + p1.handleOut.x, p1.y + p1.handleOut.y);
                const c2 = this.toCanvas(p2.x + p2.handleIn.x, p2.y + p2.handleIn.y);
                const end = this.toCanvas(p2.x, p2.y);
                ctx.bezierCurveTo(c1.x, c1.y, c2.x, c2.y, end.x, end.y);
            }

            if (s.closed && points.length > 2) {
                const p1 = points[points.length - 1];
                const p2 = points[0];
                const c1 = this.toCanvas(p1.x + p1.handleOut.x, p1.y + p1.handleOut.y);
                const c2 = this.toCanvas(p2.x + p2.handleIn.x, p2.y + p2.handleIn.y);
                const end = this.toCanvas(p2.x, p2.y);
                ctx.bezierCurveTo(c1.x, c1.y, c2.x, c2.y, end.x, end.y);
                ctx.closePath();
                ctx.fillStyle = isSelected ? "rgba(0, 204, 255, 0.15)" : "rgba(255, 255, 255, 0.05)";
                ctx.fill();
            }
            ctx.stroke();

            points.forEach((p, pIdx) => {
                const isPSelected = (this.selectedPoint?.sIdx === sIdx && this.selectedPoint?.pIdx === pIdx) ||
                    (this.selectedHandle?.sIdx === sIdx && this.selectedHandle?.pIdx === pIdx);
                const isHovered = this.hovered?.sIdx === sIdx && this.hovered?.pIdx === pIdx;

                if (isPSelected || isHovered) {
                    const cP = this.toCanvas(p.x, p.y);
                    const cIn = this.toCanvas(p.x + p.handleIn.x, p.y + p.handleIn.y);
                    const cOut = this.toCanvas(p.x + p.handleOut.x, p.y + p.handleOut.y);

                    ctx.strokeStyle = '#888';
                    ctx.setLineDash([2, 4]);
                    ctx.beginPath();
                    ctx.moveTo(cIn.x, cIn.y); ctx.lineTo(cP.x, cP.y); ctx.lineTo(cOut.x, cOut.y);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    ctx.fillStyle = '#ff0';
                    ctx.fillRect(cIn.x - 1.5 * dpr, cIn.y - 1.5 * dpr, 3 * dpr, 3 * dpr);
                    ctx.fillRect(cOut.x - 1.5 * dpr, cOut.y - 1.5 * dpr, 3 * dpr, 3 * dpr);
                }

                const cP = this.toCanvas(p.x, p.y);
                ctx.fillStyle = (this.selectedPoint?.sIdx === sIdx && this.selectedPoint?.pIdx === pIdx) ? '#0f0' : (isHovered ? '#ff0' : '#fff');
                ctx.beginPath();
                ctx.arc(cP.x, cP.y, 1.5 * dpr, 0, Math.PI * 2);
                ctx.fill();
            });
        });
    }

    destroy() {
        window.removeEventListener("mousemove", this._mouseMoveHandler);
        window.removeEventListener("mouseup", this._mouseUpHandler);
        this.resizeObserver.disconnect();
    }
}

app.registerExtension({
    name: "Detonate.RotoBezier",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "DetonateRotoBezier") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const splineW = this.widgets.find(w => w.name === "spline_data");
                if (splineW) {
                    splineW.type = "hidden";
                    splineW.computeSize = () => [0, -4];

                    const editor = new RotoBezierCanvas(this, splineW);
                    const domWidget = this.addDOMWidget("roto_canvas", "ROTO_CANVAS", editor.container, { serialize: false, hideOnZoom: false });

                    const onResize = this.onResize;
                    this.onResize = function (size) {
                        if (onResize) onResize.apply(this, arguments);
                        const canvasHeight = Math.max(200, size[1] - 150);
                        editor.container.style.height = canvasHeight + "px";
                        domWidget.size = [size[0] - 14, canvasHeight];
                    };

                    this.size[0] = 800;
                    this.size[1] = 600;

                    this.onRemoved = () => editor.destroy();
                }
                return r;
            };
        }
    }
});
