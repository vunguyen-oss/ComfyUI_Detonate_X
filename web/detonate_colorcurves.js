/**
 * ComfyUI_Detonate - ColorCurves Interactive Widget
 *
 * Professional Grade:
 * - Left-Click: Add/Drag Point
 * - Right-Click: Remove Point
 * - Monotonic Cubic Splines (Bezier-style smoothness)
 * - Input-aware Histogram (Traces links to upstream source)
 * - Unlocked 2-Axis Endpoints
 * - Preset Synchronization & Smart Override
 */

import { app } from "../../scripts/app.js";

class CurveEditorCanvas {
    constructor(node, widgetsData) {
        this.node = node;
        
        // Native string widgets
        this.masterWidget = widgetsData.master;
        this.redWidget = widgetsData.red;
        this.greenWidget = widgetsData.green;
        this.blueWidget = widgetsData.blue;
        this.presetWidget = widgetsData.preset;

        this.curves = {
            master: this.parseCurve(this.masterWidget.value),
            red: this.parseCurve(this.redWidget.value),
            green: this.parseCurve(this.greenWidget.value),
            blue: this.parseCurve(this.blueWidget.value)
        };
        
        this.colors = {
            master: '#ffffff',
            red: '#ff4444',
            green: '#44ff44',
            blue: '#4488ff'
        };

        // Unified Presets with Python
        this.PRESETS = {
            "linear": "0,0;1,1",
            "s_curve": "0,0;0.25,0.2;0.75,0.8;1,1",
            "lift_shadows": "0,0.1;0.5,0.5;1,1",
            "crush_blacks": "0,0;0.1,0;1,1",
            "filmic": "0,0.02;0.18,0.18;0.9,0.95;1,0.98",
        };

        this.currentChannel = 'master';
        this.selectedPointIdx = null;
        this.dragging = false;
        
        this.histogramData = null;
        this._histogramLastUpdate = 0;

        // Container
        this.container = document.createElement("div");
        this.container.style.width = "100%";
        this.container.style.height = "100%";
        this.container.style.boxSizing = "border-box";
        this.container.style.padding = "2px";
        this.container.style.backgroundColor = "#080808";
        this.container.style.borderTop = "1px solid #333";
        this.container.style.borderBottom = "1px solid #333";
        this.container.style.position = "relative";
        this.container.style.overflow = "hidden";
        
        // Canvas
        this.canvas = document.createElement("canvas");
        this.canvas.style.width = "100%";
        this.canvas.style.height = "100%";
        this.canvas.style.cursor = "crosshair";
        this.canvas.style.touchAction = "none";
        this.canvas.style.display = "block";
        this.canvas.tabIndex = 1; 

        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");

        // Events
        this.canvas.addEventListener("mousedown", this.onMouseDown.bind(this));
        this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
        
        this._mouseMoveHandler = this.onMouseMove.bind(this);
        this._mouseUpHandler = this.onMouseUp.bind(this);
        window.addEventListener("mousemove", this._mouseMoveHandler);
        window.addEventListener("mouseup", this._mouseUpHandler);
        
        this.canvas.addEventListener("keydown", this.onKeyDown.bind(this));

        this.resizeObserver = new ResizeObserver((entries) => {
            for (let entry of entries) {
                const rect = entry.contentRect;
                if (rect.width > 0 && rect.height > 0) {
                    this.canvas.width = Math.floor(rect.width * window.devicePixelRatio);
                    this.canvas.height = Math.floor(rect.height * window.devicePixelRatio);
                    this.render();
                }
            }
        });
        this.resizeObserver.observe(this.container);

        this._histTimer = setInterval(() => this.updateHistogram(), 2000);
        requestAnimationFrame(() => this.render());

        // Hook into preset changes if the widget exists
        if (this.presetWidget) {
            const orgCallback = this.presetWidget.callback;
            this.presetWidget.callback = (v) => {
                if (orgCallback) orgCallback.apply(this.presetWidget, [v]);
                if (v !== "none" && this.PRESETS[v]) {
                    this.applyPresetToCurves(v);
                }
            };
        }
    }

    applyPresetToCurves(presetName) {
        const curveStr = this.PRESETS[presetName];
        if (curveStr) {
            this.curves.master = this.parseCurve(curveStr);
            this.updateWidgets(false); // Don't reset preset to none
            this.render();
        }
    }

    parseCurve(curveStr) {
        if (!curveStr) return [{x: 0, y: 0}, {x: 1, y: 1}];
        try {
            const points = [];
            const pairs = curveStr.split(';');
            for (const pair of pairs) {
                if (!pair.trim()) continue;
                const [x_s, y_s] = pair.split(',');
                points.push({x: parseFloat(x_s), y: parseFloat(y_s)});
            }
            if (points.length < 2) return [{x: 0, y: 0}, {x: 1, y: 1}];
            points.sort((a,b) => a.x - b.x);
            return points;
        } catch (e) {
            return [{x: 0, y: 0}, {x: 1, y: 1}];
        }
    }

    serializeCurve(points) {
        return points.map(p => `${p.x.toFixed(3)},${p.y.toFixed(3)}`).join(';');
    }

    updateWidgets(resetPreset = true) {
        this.masterWidget.value = this.serializeCurve(this.curves.master);
        this.redWidget.value = this.serializeCurve(this.curves.red);
        this.greenWidget.value = this.serializeCurve(this.curves.green);
        this.blueWidget.value = this.serializeCurve(this.curves.blue);
        
        // Smart override: If user moves points, preset is no longer active
        if (resetPreset && this.presetWidget && this.presetWidget.value !== "none") {
            this.presetWidget.value = "none";
        }

        if (app.graph) this.node.setDirtyCanvas(true, true);
    }

    updateHistogram() {
        let img = null;
        if (this.node.inputs && this.node.inputs[0]) {
            const linkId = this.node.inputs[0].link;
            if (linkId != null && app.graph) {
                const link = app.graph.links[linkId];
                if (link) {
                    const originNode = app.graph.getNodeById(link.origin_id);
                    if (originNode) {
                        img = (originNode.imgs && originNode.imgs[0]) || originNode.image;
                    }
                }
            }
        }
        if (!img) img = (this.node.imgs && this.node.imgs[0]) || this.node.image;
        if (!img) { this.histogramData = null; return; }

        const now = Date.now();
        if (now - this._histogramLastUpdate < 2000) return;
        this._histogramLastUpdate = now;

        const tempCanvas = document.createElement("canvas");
        const tCtx = tempCanvas.getContext("2d");
        tempCanvas.width = 128; tempCanvas.height = 128;
        tCtx.drawImage(img, 0, 0, 128, 128);
        
        try {
            const data = tCtx.getImageData(0, 0, 128, 128).data;
            const hist = { r: new Array(256).fill(0), g: new Array(256).fill(0), b: new Array(256).fill(0), m: new Array(256).fill(0) };
            for (let i = 0; i < data.length; i += 4) {
                const r = data[i], g = data[i+1], b = data[i+2];
                const m = Math.round((r + g + b) / 3);
                hist.r[r]++; hist.g[g]++; hist.b[b]++; hist.m[m]++;
            }
            const max = Math.max(...hist.m);
            if (max > 0) {
                this.histogramData = {
                    r: hist.r.map(v => v / max), g: hist.g.map(v => v / max),
                    b: hist.b.map(v => v / max), m: hist.m.map(v => v / max)
                };
            }
            this.render();
        } catch (e) { }
    }

    destroy() {
        window.removeEventListener("mousemove", this._mouseMoveHandler);
        window.removeEventListener("mouseup", this._mouseUpHandler);
        clearInterval(this._histTimer);
        if (this.resizeObserver) this.resizeObserver.disconnect();
    }

    getPadding() { return 24 * window.devicePixelRatio; }

    normalizedToPixel(nx, ny) {
        const padding = this.getPadding();
        const drawW = this.canvas.width - padding * 2;
        const drawH = this.canvas.height - padding * 2;
        return {
            x: padding + nx * drawW,
            y: padding + (1.0 - ny) * drawH
        };
    }

    pixelToNormalized(px, py) {
        const padding = this.getPadding();
        const drawW = this.canvas.width - padding * 2;
        const drawH = this.canvas.height - padding * 2;
        let nx = (px * window.devicePixelRatio - padding) / drawW;
        let ny = 1.0 - ((py * window.devicePixelRatio - padding) / drawH);
        return {
            x: Math.max(0, Math.min(1, nx)),
            y: Math.max(0, Math.min(1, ny))
        };
    }

    onMouseDown(e) {
        const pos = {x: e.offsetX, y: e.offsetY};
        const nearIdx = this.findNearPoint(pos.x, pos.y);
        
        if (e.button === 0) { // Left Click -> ADD / DRAG
            if (nearIdx !== null) {
                this.selectedPointIdx = nearIdx;
                this.dragging = true;
            } else {
                if (pos.x < 20 || pos.x > (this.canvas.clientWidth - 20) || 
                    pos.y < 20 || pos.y > (this.canvas.clientHeight - 20)) return;

                const norm = this.pixelToNormalized(pos.x, pos.y);
                const arr = this.curves[this.currentChannel];
                arr.push(norm);
                arr.sort((a,b) => a.x - b.x);
                this.selectedPointIdx = arr.findIndex(p => p.x === norm.x && p.y === norm.y);
                this.dragging = true;
                this.updateWidgets();
            }
            this.render();
        } else if (e.button === 2) { // Right Click -> REMOVE
            if (nearIdx !== null) {
                const arr = this.curves[this.currentChannel];
                if (arr.length > 2) {
                    arr.splice(nearIdx, 1);
                    this.selectedPointIdx = null;
                    this.updateWidgets();
                    this.render();
                }
            }
        }
    }

    findNearPoint(px, py, threshold = 12) {
        const pts = this.curves[this.currentChannel];
        const pixelThreshold = threshold * window.devicePixelRatio;
        for (let i = 0; i < pts.length; i++) {
            const pCanvas = this.normalizedToPixel(pts[i].x, pts[i].y);
            const dist = Math.sqrt((pCanvas.x - px * window.devicePixelRatio) ** 2 + (pCanvas.y - py * window.devicePixelRatio) ** 2);
            if (dist < pixelThreshold) return i;
        }
        return null;
    }

    onMouseMove(e) {
        if (this.dragging && this.selectedPointIdx !== null) {
            const rect = this.canvas.getBoundingClientRect();
            const pos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
            const norm = this.pixelToNormalized(pos.x, pos.y);
            const arr = this.curves[this.currentChannel];
            
            const minX = this.selectedPointIdx > 0 ? arr[this.selectedPointIdx-1].x + 0.001 : 0.0;
            const maxX = this.selectedPointIdx < arr.length - 1 ? arr[this.selectedPointIdx+1].x - 0.001 : 1.0;
            
            arr[this.selectedPointIdx].x = Math.max(minX, Math.min(maxX, norm.x));
            arr[this.selectedPointIdx].y = norm.y;
            
            this.updateWidgets();
            this.render();
        }
    }

    onMouseUp(e) { this.dragging = false; }

    onKeyDown(e) {
        if (e.key === 'Delete' || e.key === 'Backspace') {
            if (this.selectedPointIdx !== null) {
                const arr = this.curves[this.currentChannel];
                if (arr.length > 2) {
                    arr.splice(this.selectedPointIdx, 1);
                    this.selectedPointIdx = null; this.updateWidgets(); this.render();
                }
            }
        }
    }

    getSplineY(pts, x) {
        if (pts.length < 2) return pts[0]?.y || 0;
        if (x <= pts[0].x) return pts[0].y;
        if (x >= pts[pts.length - 1].x) return pts[pts.length - 1].y;
        let i = 0;
        while (i < pts.length - 1 && x > pts[i+1].x) i++;
        const p0 = pts[i], p1 = pts[i+1];
        const h = p1.x - p0.x;
        if (h <= 0) return p0.y;
        const t = (x - p0.x) / h;
        if (pts.length === 2) return p0.y + t * (p1.y - p0.y);
        const tangents = new Array(pts.length).fill(0);
        const secants = new Array(pts.length - 1);
        for (let j = 0; j < pts.length - 1; j++) {
            secants[j] = (pts[j+1].y - pts[j].y) / (pts[j+1].x - pts[j].x);
        }
        tangents[0] = secants[0]; tangents[pts.length - 1] = secants[pts.length - 2];
        for (let j = 1; j < pts.length - 1; j++) tangents[j] = (secants[j-1] + secants[j]) / 2;
        for (let j = 0; j < pts.length - 1; j++) {
            if (secants[j] === 0) { tangents[j] = 0; tangents[j+1] = 0; }
            else {
                const a = tangents[j] / secants[j], b = tangents[j+1] / secants[j], d = a*a + b*b;
                if (d > 9.0) { const tau = 3.0 / Math.sqrt(d); tangents[j] = tau * a * secants[j]; tangents[j+1] = tau * b * secants[j]; }
            }
        }
        const t2 = t * t, t3 = t2 * t;
        const h00 = 2*t3 - 3*t2 + 1, h10 = t3 - 2*t2 + t, h01 = -2*t3 + 3*t2, h11 = t3 - t2;
        return h00 * p0.y + h10 * h * tangents[i] + h01 * p1.y + h11 * h * tangents[i+1];
    }

    render() {
        if (!this.ctx || this.canvas.width === 0) return;
        const w = this.canvas.width, h = this.canvas.height;
        const padding = this.getPadding();
        const drawW = w - padding * 2, drawH = h - padding * 2;
        this.ctx.clearRect(0, 0, w, h);
        this.ctx.fillStyle = '#080808';
        this.ctx.fillRect(padding, padding, drawW, drawH);
        
        if (this.histogramData) {
            this.ctx.globalAlpha = 0.25;
            const chan = this.currentChannel === 'master' ? 'm' : this.currentChannel[0];
            const data = this.histogramData[chan];
            if (data) {
                this.ctx.fillStyle = this.colors[this.currentChannel];
                const barW = drawW / 256;
                for (let i = 0; i < 256; i++) {
                    const bh = data[i] * drawH;
                    this.ctx.fillRect(padding + i * barW, h - padding - bh, barW + 1, bh);
                }
            }
            this.ctx.globalAlpha = 1.0;
        }

        this.ctx.strokeStyle = '#222'; this.ctx.lineWidth = 1 * window.devicePixelRatio; this.ctx.beginPath();
        for (let i = 1; i < 4; i++) {
            const bx = padding + (drawW * i / 4), by = padding + (drawH * i / 4);
            this.ctx.moveTo(bx, padding); this.ctx.lineTo(bx, h - padding);
            this.ctx.moveTo(padding, by); this.ctx.lineTo(w - padding, by);
        }
        this.ctx.stroke();
        this.ctx.strokeStyle = '#444'; this.ctx.setLineDash([5, 5]); this.ctx.beginPath();
        this.ctx.moveTo(padding, h - padding); this.ctx.lineTo(w - padding, padding);
        this.ctx.stroke(); this.ctx.setLineDash([]);

        this.ctx.fillStyle = '#888'; this.ctx.font = `${Math.round(10 * window.devicePixelRatio)}px monospace`;
        this.ctx.textAlign = 'right'; this.ctx.textBaseline = 'middle';
        [0, 0.5, 1].forEach(v => this.ctx.fillText(v.toFixed(1), padding - 8, this.normalizedToPixel(0, v).y));
        this.ctx.textAlign = 'center'; this.ctx.textBaseline = 'top';
        [0, 0.5, 1].forEach(v => this.ctx.fillText(v.toFixed(1), this.normalizedToPixel(v, 0).x, h - padding + 8));

        ['master', 'red', 'green', 'blue'].forEach(c => {
            if (c === this.currentChannel) return;
            this.drawChannelCurve(c, 0.45, 1.5 * window.devicePixelRatio);
        });
        this.drawChannelCurve(this.currentChannel, 1.0, 3.0 * window.devicePixelRatio);

        const cur = this.curves[this.currentChannel];
        cur.forEach((pt, i) => {
            const p = this.normalizedToPixel(pt.x, pt.y);
            const sel = (i === this.selectedPointIdx);
            this.ctx.beginPath();
            this.ctx.fillStyle = sel ? '#fff' : this.colors[this.currentChannel];
            this.ctx.arc(p.x, p.y, (sel ? 6 : 4) * window.devicePixelRatio, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.strokeStyle = '#000'; this.ctx.lineWidth = 1 * window.devicePixelRatio; this.ctx.stroke();
            if (sel) {
                this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1.5 * window.devicePixelRatio;
                this.ctx.beginPath(); this.ctx.arc(p.x, p.y, 9 * window.devicePixelRatio, 0, Math.PI * 2); this.ctx.stroke();
            }
        });
    }
    
    drawChannelCurve(channel, alpha, width) {
        const pts = this.curves[channel];
        if (!pts || pts.length < 2) return;
        this.ctx.beginPath();
        this.ctx.strokeStyle = this.colors[channel]; this.ctx.lineWidth = width; this.ctx.globalAlpha = alpha;
        this.ctx.lineJoin = "round"; this.ctx.lineCap = "round";
        const pStart = this.normalizedToPixel(0, pts[0].y);
        this.ctx.moveTo(pStart.x, pStart.y);
        const steps = 200;
        for (let i = 0; i <= steps; i++) {
            const nx = i / steps;
            const ny = this.getSplineY(pts, nx);
            const p = this.normalizedToPixel(nx, ny);
            this.ctx.lineTo(p.x, p.y);
        }
        const pEnd = this.normalizedToPixel(1, pts[pts.length - 1].y);
        this.ctx.lineTo(pEnd.x, pEnd.y);
        this.ctx.stroke();
        this.ctx.globalAlpha = 1.0;
    }
}

app.registerExtension({
    name: "Detonate.ColorCurves",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DetonateColorCurves") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const mw = this.widgets?.find(w => w.name === "master_curve");
                const rw = this.widgets?.find(w => w.name === "red_curve");
                const gw = this.widgets?.find(w => w.name === "green_curve");
                const bw = this.widgets?.find(w => w.name === "blue_curve");
                const pw = this.widgets?.find(w => w.name === "preset");
                
                if (mw && rw && gw && bw) {
                    mw.type = "hidden"; mw.computeSize = () => [0, -4];
                    rw.type = "hidden"; rw.computeSize = () => [0, -4];
                    gw.type = "hidden"; gw.computeSize = () => [0, -4];
                    bw.type = "hidden"; bw.computeSize = () => [0, -4];
                    
                    const editor = new CurveEditorCanvas(this, { 
                        master: mw, red: rw, green: gw, blue: bw, preset: pw 
                    });
                    const widget = this.addDOMWidget("curves_editor_dom", "CURVES_EDITOR_DOM", editor.container, { serialize: false, hideOnZoom: false });
                    
                    this.addWidget("combo", "Channel", "master", (v) => {
                        editor.currentChannel = v;
                        editor.selectedPointIdx = null; editor.render();
                    }, { values: ["master", "red", "green", "blue"] });
                    
                    this.addWidget("button", "Reset", null, () => {
                        if (editor.currentChannel) {
                            editor.curves[editor.currentChannel] = [{x:0, y:0}, {x:1, y:1}];
                            editor.selectedPointIdx = null; 
                            editor.updateWidgets(true); // Reset preset to none
                            editor.render();
                        }
                    });
                    
                    const onResize = this.onResize;
                    this.onResize = function(size) {
                        if (onResize) onResize.apply(this, arguments);
                        const graphHeight = Math.max(160, size[1] - 130);
                        editor.container.style.height = graphHeight + "px";
                        widget.size = [size[0] - 14, graphHeight];
                    };

                    this.size[0] = Math.max(this.size[0], 280);
                    this.size[1] = Math.max(this.size[1], 460);
                    this.onRemoved = () => editor.destroy();
                }
                return r;
            };
        }
    }
});
