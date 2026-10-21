(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);

  const LABELS = ["H", "A", "D", "X"];
  const COLORS = {
    H: "#4ea8ff",
    A: "#ff6b5f",
    D: "#ffc857",
    X: "#607286",
  };
  const NAMES = {
    H: "Barcelona",
    A: "Athletic Club",
    D: "Balón muerto",
    X: "Sin evaluar",
  };
  const TEAM_NAMES = {
    "-1": "N/A",
    0: "Barcelona",
    1: "Athletic Club",
  };
  const MONO = '"SFMono-Regular", "Cascadia Code", "Roboto Mono", monospace';

  const ui = {
    video: $("#video"),
    videoError: $("#videoError"),
    syncState: $("#syncState"),
    clock: $("#clock"),
    frameMeta: $("#frameMeta"),
    frameNumber: $("#frameNumber"),
    timecode: $("#timecode"),
    predBadge: $("#predBadge"),
    predName: $("#predName"),
    gtBadge: $("#gtBadge"),
    gtName: $("#gtName"),
    utBadge: $("#utBadge"),
    utName: $("#utName"),
    agreement: $("#agreement"),
    utAgreement: $("#utAgreement"),
    spellText: $("#spellText"),
    spellDuration: $("#spellDuration"),
    tagText: $("#tagText"),
    tagDetail: $("#tagDetail"),
    gtWindow: $("#gtWindow"),
    videoDuration: $("#videoDuration"),
    metricsCard: $("#metricsCard"),
    metrics: $("#metrics"),
    utMetrics: $("#utMetrics"),
    utStatus: $("#utStatus"),
    utAnchor: $("#utAnchor"),
    utClear: $("#utClear"),
    playBtn: $("#playBtn"),
    frameInput: $("#frameInput"),
    goFrame: $("#goFrame"),
    gotoPublic: $("#gotoPublic"),
    gotoStart: $("#gotoStart"),
    prevChange: $("#prevChange"),
    nextChange: $("#nextChange"),
    prevSpell: $("#prevSpell"),
    nextSpell: $("#nextSpell"),
    prevDiff: $("#prevDiff"),
    nextDiff: $("#nextDiff"),
    overview: $("#overview"),
    detail: $("#detail"),
    overviewHint: $("#overviewHint"),
    detailHint: $("#detailHint"),
    hoverInfo: $("#hoverInfo"),
    fatal: $("#fatal"),
    fatalMessage: $("#fatalMessage"),
  };

  const state = {
    data: null,
    video: ui.video,
    frame: -1,
    initialFrame: 0,
    scrubbing: false,
    pointerId: null,
    scrubSource: null,
    hoverFrame: null,
    hoverSource: null,
    loadedMetadata: false,
    syncWarning: false,
    lastRenderAt: 0,
    pendingSeekFrame: null,
    utAnchor: null,
    utSaving: false,
    anchorDrag: null,
  };

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  function codeAt(text, frame) {
    if (!text) return null;
    return text.charAt(frame);
  }

  function codeClass(code) {
    return LABELS.includes(code) ? code : "X";
  }

  function setBadge(element, code) {
    const c = codeClass(code);
    element.className = `badge badge-${c}`;
    element.textContent = code || "—";
  }

  function setSync(text, kind) {
    ui.syncState.textContent = text;
    ui.syncState.className = `status-pill ${kind || ""}`.trim();
  }

  function pad(value, length) {
    return String(value).padStart(length, "0");
  }

  function formatTime(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "—";
    const totalMs = Math.round(seconds * 1000);
    const hours = Math.floor(totalMs / 3_600_000);
    const minutes = Math.floor((totalMs % 3_600_000) / 60_000);
    const secs = Math.floor((totalMs % 60_000) / 1000);
    const millis = totalMs % 1000;
    return `${hours}:${pad(minutes, 2)}:${pad(secs, 2)}.${pad(millis, 3)}`;
  }

  function formatShort(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "—";
    const total = Math.floor(seconds);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    if (hours > 0) return `${hours}:${pad(minutes, 2)}:${pad(secs, 2)}`;
    return `${minutes}:${pad(secs, 2)}`;
  }

  function formatFrame(frame) {
    return frame.toLocaleString("es-ES");
  }

  function frameFromTime(seconds) {
    if (!state.data) return 0;
    const raw = Math.floor(seconds * state.data.fps + 1e-6);
    return clamp(raw, 0, state.data.n_frames - 1);
  }

  function timeForFrame(frame) {
    return (frame + 0.5) / state.data.fps;
  }

  function lowerBound(values, target) {
    let lo = 0;
    let hi = values.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (values[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function findInterval(intervals, frame) {
    if (!intervals || intervals.length === 0) return null;
    let lo = 0;
    let hi = intervals.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (intervals[mid][0] <= frame) lo = mid + 1;
      else hi = mid;
    }
    const candidate = intervals[lo - 1];
    return candidate && frame >= candidate[0] && frame < candidate[1] ? candidate : null;
  }

  function previousValue(values, frame) {
    if (!values || values.length === 0) return null;
    const index = lowerBound(values, frame);
    return index > 0 ? values[index - 1] : null;
  }

  function nextValue(values, frame) {
    if (!values || values.length === 0) return null;
    const index = lowerBound(values, frame + 1);
    return index < values.length ? values[index] : null;
  }

  function findTag(frame) {
    const tags = state.data ? state.data.tags : null;
    if (!tags) return null;
    for (const tag of tags) {
      if (tag.start <= frame && frame < tag.stop) return tag;
    }
    return null;
  }

  function utCodeAt(frame) {
    const segment = findInterval(state.data ? state.data.ut_segments : null, frame);
    return segment ? LABELS[segment[2]] : "X";
  }

  function metricText(value) {
    if (typeof value !== "number") return String(value);
    return Number.isInteger(value) ? value.toLocaleString("es-ES") : value.toFixed(3);
  }

  function updateUTStatus(prefix) {
    const data = state.data;
    if (!data) return;
    const frames = Number(data.ut_labeled_frames) || 0;
    let anchor = "";
    if (state.utAnchor !== null) {
      anchor = state.utAnchor >= data.n_frames ? " · ancla al final" : ` · ancla en ${formatFrame(state.utAnchor)}`;
    }
    ui.utStatus.textContent = `${prefix || ""}UT: ${formatFrame(frames)} frames etiquetados${anchor}`;
  }

  function utAnchorFrames() {
    const frames = new Set();
    for (const [start, stop] of state.data ? state.data.ut_segments : []) {
      frames.add(start);
      frames.add(stop);
    }
    return Array.from(frames).sort((a, b) => a - b);
  }

  function frameToX(canvas, source, frame) {
    const rect = canvas.getBoundingClientRect();
    if (source === "overview") return (frame / state.data.n_frames) * rect.width;
    const { start, stop } = detailWindow();
    return ((frame - start) / Math.max(1, stop - start)) * rect.width;
  }

  function findDraggableAnchor(clientX, clientY, source) {
    if (!state.data) return null;
    const canvas = source === "overview" ? ui.overview : ui.detail;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0) return null;
    const top = source === "overview" ? 54 : 82;
    const bottom = source === "overview" ? 82 : 118;
    const y = clientY - rect.top;
    if (y < top || y > bottom) return null;
    const x = clientX - rect.left;
    let best = null;
    let bestDistance = 7;
    for (const frame of utAnchorFrames()) {
      const distance = Math.abs(frameToX(canvas, source, frame) - x);
      if (distance <= bestDistance) {
        best = { frame, pending: false };
        bestDistance = distance;
      }
    }
    if (state.utAnchor !== null) {
      const distance = Math.abs(frameToX(canvas, source, state.utAnchor) - x);
      if (distance < bestDistance) best = { frame: state.utAnchor, pending: true };
    }
    return best;
  }

  function drawAnchors(ctx, xOf, top, bottom, width) {
    const frames = utAnchorFrames();
    ctx.save();
    for (const frame of frames) {
      const x = xOf(frame);
      if (x < -4 || x > width + 4) continue;
      ctx.strokeStyle = "rgba(232,240,247,0.85)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, top);
      ctx.lineTo(x + 0.5, bottom);
      ctx.stroke();
      ctx.fillStyle = "#e8f0f7";
      ctx.beginPath();
      ctx.moveTo(x - 3, top - 5);
      ctx.lineTo(x + 3, top - 5);
      ctx.lineTo(x, top);
      ctx.closePath();
      ctx.fill();
    }

    const pending = state.utAnchor;
    if (pending !== null && !frames.includes(pending)) {
      const x = xOf(pending);
      if (x >= -4 && x <= width + 4) {
        ctx.strokeStyle = "rgba(255,200,87,0.9)";
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x + 0.5, top);
        ctx.lineTo(x + 0.5, bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#ffc857";
        ctx.beginPath();
        ctx.moveTo(x - 3, top - 5);
        ctx.lineTo(x + 3, top - 5);
        ctx.lineTo(x, top);
        ctx.closePath();
        ctx.fill();
      }
    }

    if (state.anchorDrag && state.anchorDrag.from !== state.anchorDrag.to) {
      const x = xOf(state.anchorDrag.to);
      if (x >= -4 && x <= width + 4) {
        ctx.strokeStyle = "rgba(255,200,87,0.95)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(x + 0.5, top);
        ctx.lineTo(x + 0.5, bottom);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  function spellStarts() {
    return state.data.pred_spells.map((spell) => spell[0]);
  }

  function groupConsecutive(values) {
    const runs = [];
    for (const value of values || []) {
      const last = runs[runs.length - 1];
      if (last && value === last[1]) last[1] = value + 1;
      else runs.push([value, value + 1]);
    }
    return runs;
  }

  function updateURL(frame) {
    if (!Number.isFinite(frame)) return;
    const url = new URL(window.location.href);
    url.searchParams.set("frame", String(frame));
    window.history.replaceState(null, "", url);
  }

  function isInsidePublic(frame) {
    return Boolean(state.data) && frame >= state.data.public[0] && frame < state.data.public[1];
  }

  function updateUI() {
    const data = state.data;
    if (!data || state.frame < 0) return;
    const frame = state.frame;
    const pred = codeAt(data.labels, frame);
    const gt = codeAt(data.gt, frame);
    const inside = isInsidePublic(frame) && gt !== null && gt !== "X";
    const seconds = frame / data.fps;

    ui.frameNumber.textContent = formatFrame(frame);
    ui.timecode.textContent = `${formatTime(seconds)} · ${seconds.toFixed(3)} s`;
    ui.clock.textContent = formatTime(seconds);
    ui.frameMeta.textContent = `frame ${formatFrame(frame)} / ${formatFrame(data.n_frames - 1)}`;

    setBadge(ui.predBadge, pred);
    ui.predName.textContent = NAMES[codeClass(pred)] || "—";

    if (data.gt_available && gt !== null) {
      setBadge(ui.gtBadge, gt);
      ui.gtName.textContent = NAMES[codeClass(gt)] || "—";
    } else {
      setBadge(ui.gtBadge, "X");
      ui.gtName.textContent = data.gt_available ? "—" : "GT no cargada";
    }

    if (!data.gt_available) {
      ui.agreement.textContent = "GT pública no cargada";
      ui.agreement.className = "agreement na";
    } else if (!inside) {
      ui.agreement.textContent = "GT pública no disponible en este frame";
      ui.agreement.className = "agreement na";
    } else if (pred === gt) {
      ui.agreement.textContent = "✓ coincide con la GT pública";
      ui.agreement.className = "agreement ok";
    } else {
      ui.agreement.textContent = `✗ no coincide · GT ${gt} · pred ${pred}`;
      ui.agreement.className = "agreement bad";
    }

    const ut = utCodeAt(frame);
    setBadge(ui.utBadge, ut);
    ui.utName.textContent = ut === "X" ? "Sin etiqueta" : NAMES[ut] || "—";
    if (ut === "X") {
      ui.utAgreement.textContent = "UT sin etiqueta en este frame";
      ui.utAgreement.className = "agreement na";
    } else if (pred === ut) {
      ui.utAgreement.textContent = "✓ coincide con la UT";
      ui.utAgreement.className = "agreement ok";
    } else {
      ui.utAgreement.textContent = `✗ no coincide · UT ${ut} · pred ${pred}`;
      ui.utAgreement.className = "agreement bad";
    }

    const spell = findInterval(data.pred_spells, frame);
    if (spell) {
      ui.spellText.textContent = `${spell[2]} · ${formatFrame(spell[0])}–${formatFrame(spell[1] - 1)}`;
      ui.spellDuration.textContent = `${((spell[1] - spell[0]) / data.fps).toFixed(2)} s`;
    } else {
      ui.spellText.textContent = "Fuera de una posesión H/A";
      ui.spellDuration.textContent = "—";
    }

    const tag = findTag(frame);
    if (tag) {
      const team = TEAM_NAMES[String(tag.team)] || "N/A";
      ui.tagText.textContent = `${tag.code || tag.type || "tag"} · ${team}`;
      const extras = Object.entries(tag.labels || {})
        .filter(([key]) => key !== "Team")
        .map(([key, value]) => `${key}: ${value}`)
        .join(" · ");
      ui.tagDetail.textContent = extras || tag.type || "—";
    } else {
      ui.tagText.textContent = "—";
      ui.tagDetail.textContent = "Sin smart tag en este frame";
    }

    if (document.activeElement !== ui.frameInput) {
      ui.frameInput.value = String(frame);
    }

    if (data.metrics) {
      const m = data.metrics;
      ui.metricsCard.classList.remove("hidden");
      ui.metrics.innerHTML = [
        ["Spell F1", m.spell_f1],
        ["Macro F1", m.macro_f1],
        ["D-F1", m.d_f1],
        ["Accuracy", m.accuracy],
        ["Spells pred.", m.n_pred_spells],
        ["Spells GT", m.n_gt_spells],
      ].map(([label, value]) => `<div>${label} <strong>${metricText(value)}</strong></div>`).join("");
    }

    const utMetrics = data.ut_metrics;
    if (utMetrics) {
      ui.utMetrics.innerHTML = [
        ["Spell F1", utMetrics.spell_f1],
        ["Macro F1", utMetrics.macro_f1],
        ["D-F1", utMetrics.d_f1],
        ["Accuracy", utMetrics.accuracy],
        ["Frames UT", data.ut_labeled_frames],
        ["Spells UT", utMetrics.n_gt_spells],
        ["Spells pred.", utMetrics.n_pred_spells],
      ].map(([label, value]) => `<div>${label} <strong>${metricText(value)}</strong></div>`).join("");
    } else {
      ui.utMetrics.textContent = "Sin etiquetas UT.";
    }
  }

  function drawTimelines() {
    if (!state.data) return;
    drawOverview();
    drawDetail();
  }

  function fitCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(rect.width * dpr);
    const height = Math.round(rect.height * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
  }

  function paintSegments(ctx, segments, xOf, y, height, width, alpha = 1) {
    if (!segments || segments.length === 0) return;
    ctx.save();
    ctx.globalAlpha = alpha;
    for (const [start, stop, code] of segments) {
      const x1 = xOf(start);
      const x2 = xOf(stop);
      if (x2 < 0 || x1 > width) continue;
      const left = Math.max(0, x1);
      const right = Math.min(width, x2);
      if (right <= left) continue;
      ctx.fillStyle = COLORS[LABELS[code]] || COLORS.X;
      ctx.fillRect(left, y, Math.max(1, right - left), height);
    }
    ctx.restore();
  }

  function drawText(ctx, text, x, y, options = {}) {
    const {
      size = 10,
      weight = 600,
      color = "#8ca0b4",
      align = "left",
      baseline = "middle",
    } = options;
    ctx.save();
    ctx.font = `${weight} ${size}px ${MONO}`;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  function drawOverview() {
    const fitted = fitCanvas(ui.overview);
    if (!fitted) return;
    const { ctx, width, height } = fitted;
    const data = state.data;
    const n = data.n_frames;
    const xOf = (frame) => (frame / n) * width;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#081018";
    ctx.fillRect(0, 0, width, height);

    for (const [start, stop] of data.halves) {
      ctx.fillStyle = "rgba(255,255,255,0.025)";
      ctx.fillRect(xOf(start), 0, Math.max(1, xOf(stop) - xOf(start)), height);
    }

    const publicX1 = xOf(data.public[0]);
    const publicX2 = xOf(data.public[1]);
    ctx.fillStyle = "rgba(78,168,255,0.08)";
    ctx.fillRect(publicX1, 0, Math.max(1, publicX2 - publicX1), height);

    paintSegments(ctx, data.pred_segments, xOf, 12, 22, width);
    if (data.gt_available) {
      paintSegments(ctx, data.gt_segments, xOf, 40, 16, width, 0.9);
    } else {
      ctx.fillStyle = "rgba(96,114,134,0.18)";
      ctx.fillRect(0, 40, width, 16);
      drawText(ctx, "GT no cargada", 8, 48, { color: "#6d8093", size: 9 });
    }
    ctx.fillStyle = "rgba(96,114,134,0.12)";
    ctx.fillRect(0, 60, width, 16);
    paintSegments(ctx, data.ut_segments, xOf, 60, 16, width);
    drawAnchors(ctx, xOf, 60, 76, width);

    drawText(ctx, "PRED", 6, 8, { size: 8, color: "#8ca0b4" });
    drawText(ctx, "GT", 6, 36, { size: 8, color: "#8ca0b4" });
    drawText(ctx, "UT", 6, 56, { size: 8, color: "#8ca0b4" });

    const tickFrames = 15 * 60 * data.fps;
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 1;
    for (let frame = 0; frame <= n; frame += tickFrames) {
      const x = xOf(frame);
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
      if (frame < n) {
        drawText(ctx, formatShort(frame / data.fps), x + 4, height - 10, {
          size: 8,
          color: "#607286",
        });
      }
    }

    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(78,168,255,0.55)";
    for (const x of [publicX1, publicX2]) {
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
    }
    ctx.restore();

    if (publicX2 - publicX1 > 120) {
      drawText(ctx, "GT pública", publicX1 + 6, 88, {
        size: 8,
        color: "rgba(168,207,243,0.8)",
      });
    }

    if (state.hoverFrame !== null && state.hoverSource === "overview") {
      const hx = xOf(state.hoverFrame);
      ctx.strokeStyle = "rgba(255,255,255,0.32)";
      ctx.beginPath();
      ctx.moveTo(hx + 0.5, 0);
      ctx.lineTo(hx + 0.5, height);
      ctx.stroke();
    }

    const x = xOf(state.frame);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, height);
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(x - 4, 0);
    ctx.lineTo(x + 4, 0);
    ctx.lineTo(x, 6);
    ctx.closePath();
    ctx.fill();
  }

  function detailWindow() {
    const data = state.data;
    const span = 40 * data.fps;
    let start = state.frame - Math.floor(span / 2);
    start = clamp(start, 0, Math.max(0, data.n_frames - span));
    return { start, stop: Math.min(data.n_frames, start + span) };
  }

  function drawDetail() {
    const fitted = fitCanvas(ui.detail);
    if (!fitted) return;
    const { ctx, width, height } = fitted;
    const data = state.data;
    const { start, stop } = detailWindow();
    const span = Math.max(1, stop - start);
    const xOf = (frame) => ((frame - start) / span) * width;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#081018";
    ctx.fillRect(0, 0, width, height);

    for (const [halfStart, halfStop] of data.halves) {
      const x1 = xOf(halfStart);
      const x2 = xOf(halfStop);
      if (x2 < 0 || x1 > width) continue;
      ctx.fillStyle = "rgba(255,255,255,0.025)";
      ctx.fillRect(Math.max(0, x1), 0, Math.max(1, Math.min(width, x2) - Math.max(0, x1)), height);
    }

    const publicX1 = xOf(data.public[0]);
    const publicX2 = xOf(data.public[1]);
    if (publicX2 >= 0 && publicX1 <= width) {
      ctx.fillStyle = "rgba(78,168,255,0.08)";
      ctx.fillRect(
        Math.max(0, publicX1),
        0,
        Math.max(1, Math.min(width, publicX2) - Math.max(0, publicX1)),
        height,
      );
    }

    const secondsStart = Math.ceil(start / data.fps);
    const secondsStop = Math.floor(stop / data.fps);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    for (let second = secondsStart; second <= secondsStop; second++) {
      const frame = second * data.fps;
      const x = xOf(frame);
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
      if (second % 5 === 0) {
        drawText(ctx, formatShort(second), x + 3, height - 9, {
          size: 8,
          color: "#607286",
        });
      }
    }

    paintSegments(ctx, data.pred_segments, xOf, 10, 30, width);
    if (data.gt_available) {
      paintSegments(ctx, data.gt_segments, xOf, 52, 24, width, 0.9);
    } else {
      ctx.fillStyle = "rgba(96,114,134,0.18)";
      ctx.fillRect(0, 52, width, 24);
    }
    ctx.fillStyle = "rgba(96,114,134,0.12)";
    ctx.fillRect(0, 88, width, 24);
    paintSegments(ctx, data.ut_segments, xOf, 88, 24, width);
    drawAnchors(ctx, xOf, 88, 112, width);

    drawText(ctx, "PRED", 6, 18, { size: 8, color: "rgba(232,240,247,0.72)" });
    drawText(ctx, data.gt_available ? "GT" : "GT no cargada", 6, 60, {
      size: 8,
      color: data.gt_available ? "rgba(232,240,247,0.72)" : "#6d8093",
    });
    drawText(ctx, "UT", 6, 96, { size: 8, color: "rgba(232,240,247,0.72)" });

    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(78,168,255,0.55)";
    for (const x of [publicX1, publicX2]) {
      if (x < 0 || x > width) continue;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
    }
    ctx.restore();

    if (state.hoverFrame !== null && state.hoverSource === "detail") {
      const hx = xOf(state.hoverFrame);
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx + 0.5, 0);
      ctx.lineTo(hx + 0.5, height);
      ctx.stroke();
    }

    const x = xOf(state.frame);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, height);
    ctx.stroke();

    drawText(
      ctx,
      `frame ${formatFrame(state.frame)} · ${formatTime(state.frame / data.fps)}`,
      width - 6,
      10,
      { size: 9, align: "right", color: "rgba(232,240,247,0.72)" },
    );
  }

  function renderFrame(frame, { force = false } = {}) {
    if (!state.data) return;
    const next = clamp(Math.round(frame), 0, state.data.n_frames - 1);
    const changed = next !== state.frame;
    state.frame = next;
    updateUI();
    if (changed || force) drawTimelines();
  }

  function renderFromTime(seconds) {
    if (!state.data || state.scrubbing || !Number.isFinite(seconds)) return;
    const next = frameFromTime(seconds);
    if (next !== state.frame) {
      renderFrame(next);
    }
  }

  function seekToFrame(frame, { pause = false, updateHistory = true } = {}) {
    if (!state.data) return;
    const target = clamp(Math.round(frame), 0, state.data.n_frames - 1);
    if (pause) state.video.pause();
    if (state.video.readyState >= 1) {
      const seconds = timeForFrame(target);
      if (Math.abs(state.video.currentTime - seconds) > 0.0005) {
        try {
          state.video.currentTime = seconds;
        } catch (error) {
          // The video element can reject a seek while metadata is still settling.
        }
      }
    } else {
      state.pendingSeekFrame = target;
    }
    renderFrame(target, { force: true });
    if (updateHistory) updateURL(target);
  }

  function jump(delta) {
    seekToFrame(state.frame + delta, { pause: true });
  }

  function jumpToValue(value) {
    if (value === null || value === undefined) return;
    seekToFrame(value, { pause: true });
  }

  function updateHover(frame, source) {
    if (!state.data) return;
    state.hoverFrame = frame;
    state.hoverSource = source;
    if (frame === null) {
      ui.hoverInfo.textContent = "—";
      return;
    }
    const pred = codeAt(state.data.labels, frame);
    const gt = codeAt(state.data.gt, frame);
    const ut = utCodeAt(frame);
    const gtText = gt !== null && gt !== "X" ? ` · GT ${gt}` : "";
    const utText = ut !== "X" ? ` · UT ${ut}` : "";
    ui.hoverInfo.textContent = `frame ${formatFrame(frame)} · ${formatTime(frame / state.data.fps)} · pred ${pred || "—"}${gtText}${utText}`;
  }

  function eventFrame(event, source) {
    const canvas = source === "overview" ? ui.overview : ui.detail;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0) return null;
    const fraction = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    if (source === "overview") {
      return clamp(Math.floor(fraction * state.data.n_frames), 0, state.data.n_frames - 1);
    }
    const { start, stop } = detailWindow();
    return clamp(start + Math.floor(fraction * (stop - start)), 0, state.data.n_frames - 1);
  }

  function attachScrubber(canvas, source) {
    canvas.addEventListener("pointerdown", (event) => {
      if (!state.data) return;
      event.preventDefault();
      state.pointerId = event.pointerId;
      canvas.setPointerCapture(event.pointerId);
      const hit = findDraggableAnchor(event.clientX, event.clientY, source);
      if (hit) {
        state.video.pause();
        state.anchorDrag = { from: hit.frame, to: hit.frame, pending: hit.pending };
        updateHover(hit.frame, source);
        drawTimelines();
        return;
      }
      state.scrubbing = true;
      state.scrubSource = source;
      const frame = eventFrame(event, source);
      if (frame !== null) {
        updateHover(frame, source);
        seekToFrame(frame, { pause: true, updateHistory: false });
      }
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!state.data) return;
      const frame = eventFrame(event, source);
      if (frame === null) return;
      if (state.anchorDrag && event.pointerId === state.pointerId) {
        state.anchorDrag.to = frame;
        updateHover(frame, source);
        drawTimelines();
      } else if (state.scrubbing && event.pointerId === state.pointerId) {
        updateHover(frame, source);
        seekToFrame(frame, { pause: true, updateHistory: false });
      } else if (!state.scrubbing) {
        canvas.style.cursor = findDraggableAnchor(event.clientX, event.clientY, source) ? "ew-resize" : "crosshair";
        updateHover(frame, source);
        if (source === "overview") drawOverview();
        else drawDetail();
      }
    });

    const finish = (event) => {
      if (event.pointerId !== state.pointerId) return;
      const drag = state.anchorDrag;
      const wasScrubbing = state.scrubbing;
      state.anchorDrag = null;
      state.scrubbing = false;
      state.pointerId = null;
      state.scrubSource = null;
      try {
        canvas.releasePointerCapture(event.pointerId);
      } catch (error) {
        // Pointer capture may already have been released by the browser.
      }
      if (drag) {
        if (drag.pending) {
          state.utAnchor = clamp(drag.to, 0, state.data.n_frames);
          updateUTStatus("Ancla movida. ");
          drawTimelines();
        } else if (drag.to !== drag.from) {
          drawTimelines();
          postUserTruth({ boundary: drag.from, to: drag.to });
        } else {
          drawTimelines();
        }
        return;
      }
      if (wasScrubbing) updateURL(state.frame);
    };
    canvas.addEventListener("pointerup", finish);
    canvas.addEventListener("pointercancel", finish);

    canvas.addEventListener("pointerleave", () => {
      if (state.scrubbing || state.anchorDrag) return;
      canvas.style.cursor = "crosshair";
      state.hoverFrame = null;
      state.hoverSource = null;
      updateHover(null, null);
      drawOverview();
      drawDetail();
    });
  }

  function setPlayButton() {
    ui.playBtn.textContent = state.video.paused ? "Reproducir" : "Pausa";
  }

  function togglePlay() {
    if (state.video.paused) {
      const promise = state.video.play();
      if (promise && typeof promise.catch === "function") {
        promise.catch(() => {
          ui.videoError.textContent = "El navegador bloqueó la reproducción. Pulsa Reproducir de nuevo.";
          ui.videoError.classList.remove("hidden");
        });
      }
    } else {
      state.video.pause();
    }
    setPlayButton();
  }

  function showFatal(message) {
    ui.fatalMessage.textContent = message;
    ui.fatal.classList.remove("hidden");
    setSync("error", "error");
  }

  async function postUserTruth(body) {
    if (!state.data || state.utSaving) return;
    state.utSaving = true;
    try {
      const response = await fetch("/api/user-truth", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status} al guardar la UT`);
      Object.assign(state.data, await response.json());
      if (body.clear || body.boundary !== undefined) {
        state.utAnchor = null;
      } else {
        state.utAnchor = clamp(body.stop, 0, state.data.n_frames);
      }
      drawTimelines();
      updateUI();
      updateUTStatus(
        body.clear ? "UT borrada. " : body.boundary !== undefined ? "Ancla movida. " : "Ancla añadida. ",
      );
    } catch (error) {
      ui.utStatus.textContent = `No se pudo guardar la UT: ${error instanceof Error ? error.message : String(error)}`;
    } finally {
      state.utSaving = false;
    }
  }

  function paintUserTruth(label) {
    if (!state.data || state.utSaving) return;
    const frame = state.frame;
    const anchor = state.utAnchor;
    const start = anchor === null ? frame : Math.min(anchor, frame);
    const stop = (anchor === null ? frame : Math.max(anchor, frame)) + 1;
    postUserTruth({ start, stop, label });
  }

  function toggleUserTruthAnchor() {
    if (!state.data) return;
    state.utAnchor = state.utAnchor === state.frame ? null : state.frame;
    updateUTStatus(state.utAnchor === null ? "Ancla quitada. " : "Ancla fijada. ");
  }

  function setup() {
    const data = state.data;
    document.title = `Possession Review · ${data.game}`;
    ui.frameInput.max = String(data.n_frames - 1);
    ui.videoDuration.textContent = `duración ${formatTime(data.duration_s)}`;

    const publicStart = data.public[0];
    const publicStop = data.public[1];
    ui.gtWindow.textContent = data.gt_available
      ? `${formatTime(publicStart / data.fps)} – ${formatTime(publicStop / data.fps)} · ${formatFrame(publicStop - publicStart)} frames`
      : "No disponible";

    const params = new URLSearchParams(window.location.search);
    const requested = Number.parseInt(params.get("frame") || "", 10);
    state.initialFrame = Number.isFinite(requested)
      ? clamp(requested, 0, data.n_frames - 1)
      : clamp(Number(data.start_frame) || publicStart, 0, data.n_frames - 1);
    state.frame = -1;
    state.pendingSeekFrame = state.initialFrame;
    renderFrame(state.initialFrame, { force: true });

    const disagreementStarts = groupConsecutive(data.disagreements).map((run) => run[0]);
    ui.prevDiff.disabled = !data.gt_available || disagreementStarts.length === 0;
    ui.nextDiff.disabled = !data.gt_available || disagreementStarts.length === 0;
    ui.gotoPublic.disabled = !data.gt_available;

    attachScrubber(ui.overview, "overview");
    attachScrubber(ui.detail, "detail");

    state.video.addEventListener("loadedmetadata", () => {
      state.loadedMetadata = true;
      const duration = state.video.duration;
      const durationText = Number.isFinite(duration) ? formatTime(duration) : "—";
      ui.videoDuration.textContent = `duración ${durationText} · ${data.fps} fps`;
      if (Number.isFinite(duration) && Math.abs(duration - data.duration_s) > 0.15) {
        state.syncWarning = true;
        setSync("duración distinta", "warn");
      } else {
        setSync("video sincronizado", "ok");
      }
      const target = state.pendingSeekFrame === null ? state.initialFrame : state.pendingSeekFrame;
      state.pendingSeekFrame = null;
      seekToFrame(target, { pause: true, updateHistory: false });
      updateURL(target);
      setPlayButton();
    });

    state.video.addEventListener("play", setPlayButton);
    state.video.addEventListener("pause", setPlayButton);
    state.video.addEventListener("ended", setPlayButton);
    state.video.addEventListener("timeupdate", () => renderFromTime(state.video.currentTime));
    state.video.addEventListener("seeked", () => {
      if (!state.scrubbing) renderFromTime(state.video.currentTime);
    });
    state.video.addEventListener("error", () => {
      const code = state.video.error ? state.video.error.code : "?";
      ui.videoError.textContent = `No se pudo reproducir el MP4 táctico (código ${code}). Comprueba que el servidor y el archivo existen.`;
      ui.videoError.classList.remove("hidden");
      setSync("error de video", "error");
    });

    if (state.video.requestVideoFrameCallback) {
      const onVideoFrame = (now, metadata) => {
        if (!state.scrubbing) renderFromTime(metadata.mediaTime);
        state.video.requestVideoFrameCallback(onVideoFrame);
      };
      state.video.requestVideoFrameCallback(onVideoFrame);
    }

    ui.playBtn.addEventListener("click", togglePlay);
    for (const button of document.querySelectorAll("[data-seek]")) {
      button.addEventListener("click", () => jump(Number(button.dataset.seek)));
    }
    ui.prevChange.addEventListener("click", () => jumpToValue(previousValue(data.changes, state.frame)));
    ui.nextChange.addEventListener("click", () => jumpToValue(nextValue(data.changes, state.frame)));
    ui.prevSpell.addEventListener("click", () => jumpToValue(previousValue(spellStarts(), state.frame)));
    ui.nextSpell.addEventListener("click", () => jumpToValue(nextValue(spellStarts(), state.frame)));
    ui.prevDiff.addEventListener("click", () => jumpToValue(data.gt_available ? previousValue(disagreementStarts, state.frame) : null));
    ui.nextDiff.addEventListener("click", () => jumpToValue(data.gt_available ? nextValue(disagreementStarts, state.frame) : null));
    ui.gotoPublic.addEventListener("click", () => seekToFrame(publicStart, { pause: true }));
    ui.gotoStart.addEventListener("click", () => seekToFrame(0, { pause: true }));
    ui.goFrame.addEventListener("click", () => {
      const value = Number.parseInt(ui.frameInput.value, 10);
      if (Number.isFinite(value)) seekToFrame(value, { pause: true });
    });
    ui.frameInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        ui.goFrame.click();
      }
    });

    for (const button of document.querySelectorAll("[data-ut-label]")) {
      button.addEventListener("click", () => paintUserTruth(button.dataset.utLabel));
    }
    ui.utAnchor.addEventListener("click", toggleUserTruthAnchor);
    ui.utClear.addEventListener("click", () => postUserTruth({ clear: true }));

    window.addEventListener("keydown", (event) => {
      const target = event.target;
      const typing =
        target instanceof Element && target.closest("textarea, select, input:not([type='number'])");
      if (!typing && !event.ctrlKey && !event.metaKey && !event.altKey) {
        const key = event.key.length === 1 ? event.key.toUpperCase() : "";
        if (!event.repeat && ["H", "A", "D", "X"].includes(key)) {
          event.preventDefault();
          paintUserTruth(key);
          return;
        }
      }
      if (target instanceof Element && target.closest("input, textarea, select, button")) return;
      if (event.code === "Space") {
        event.preventDefault();
        togglePlay();
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        const delta = (event.shiftKey ? data.fps : 1) * (event.key === "ArrowLeft" ? -1 : 1);
        jump(delta);
      } else if (event.key === "[") {
        event.preventDefault();
        jumpToValue(previousValue(data.changes, state.frame));
      } else if (event.key === "]") {
        event.preventDefault();
        jumpToValue(nextValue(data.changes, state.frame));
      } else if (event.key === ",") {
        event.preventDefault();
        jumpToValue(previousValue(spellStarts(), state.frame));
      } else if (event.key === ".") {
        event.preventDefault();
        jumpToValue(nextValue(spellStarts(), state.frame));
      } else if (event.key === "n") {
        event.preventDefault();
        jumpToValue(data.gt_available ? nextValue(disagreementStarts, state.frame) : null);
      } else if (event.key === "N") {
        event.preventDefault();
        jumpToValue(data.gt_available ? previousValue(disagreementStarts, state.frame) : null);
      } else if (event.key.toLowerCase() === "g" && data.gt_available) {
        event.preventDefault();
        seekToFrame(publicStart, { pause: true });
      } else if (event.key === "Home") {
        event.preventDefault();
        seekToFrame(0, { pause: true });
      } else if (event.key === "End") {
        event.preventDefault();
        seekToFrame(data.n_frames - 1, { pause: true });
      }
    });

    const resize = () => drawTimelines();
    if (window.ResizeObserver) {
      const observer = new ResizeObserver(resize);
      observer.observe(ui.overview.parentElement);
      observer.observe(ui.detail.parentElement);
    }
    window.addEventListener("resize", resize);

    ui.video.src = "/video";
    ui.video.load();
    setPlayButton();
    drawTimelines();
    updateUI();
    updateUTStatus();
  }

  async function boot() {
    try {
      const response = await fetch("/api/viewer-data", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status} al cargar /api/viewer-data`);
      state.data = await response.json();
      if (!state.data || !Array.isArray(state.data.pred_spells)) {
        throw new Error("El payload del visor no tiene el formato esperado.");
      }
      setup();
    } catch (error) {
      showFatal(error instanceof Error ? error.message : String(error));
    }
  }

  boot();
})();
