/* Circuit Siege client. The server is authoritative: this file only renders
   state, sends the human's chosen cell, and asks the server for the AI's move. */
"use strict";

const SIZE = 7;
const CORES = new Set([10, 22, 24, 26, 38]);
const COLS = "ABCDEFG";
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const cellName = (i) => `${COLS[i % SIZE]}${Math.floor(i / SIZE) + 1}`;
const cellValue = (i) => (CORES.has(i) ? 3 : 1);
const pct = (p) => Math.round(p * 100);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const ui = {
  game: null,
  busy: false,
  settings: { difficulty: "normal", first: "human" },
  lastAi: null,          // last AI event (with all candidate ratings)
  status: null,
};

/* ---------------- API ---------------- */
async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Can't reach the game server. Is run_game.py still running?");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

/* ---------------- board construction ---------------- */
function geometry() {
  // Matches CSS: padding 4% and gap 2.2% of the board width.
  const pad = 4, gap = 2.2;
  const cell = (100 - 2 * pad - (SIZE - 1) * gap) / SIZE;
  const centre = (k) => pad + k * (cell + gap) + cell / 2;
  return { pad, gap, cell, centre };
}

function buildTraces() {
  const { centre, pad } = geometry();
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "traces");
  svg.setAttribute("viewBox", "0 0 100 100");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("aria-hidden", "true");
  const add = (tag, attrs) => {
    const el = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    svg.appendChild(el);
    return el;
  };
  for (let k = 0; k < SIZE; k++) {
    add("line", { x1: centre(0), y1: centre(k), x2: centre(SIZE - 1), y2: centre(k) });
    add("line", { x1: centre(k), y1: centre(0), x2: centre(k), y2: centre(SIZE - 1) });
  }
  // vias where diagonals cross between pads
  for (let r = 0; r < SIZE - 1; r++) for (let c = 0; c < SIZE - 1; c++) {
    if ((r * 3 + c * 5) % 4 === 0) {
      add("circle", { class: "via", cx: (centre(c) + centre(c + 1)) / 2, cy: (centre(r) + centre(r + 1)) / 2, r: 0.9 });
    }
  }
  // silkscreen coordinates
  for (let k = 0; k < SIZE; k++) {
    const t1 = add("text", { class: "silk", x: centre(k), y: pad * 0.62, "text-anchor": "middle", "font-size": 2.3 });
    t1.textContent = COLS[k];
    const t2 = add("text", { class: "silk", x: pad * 0.5, y: centre(k) + 0.8, "text-anchor": "middle", "font-size": 2.3 });
    t2.textContent = String(k + 1);
  }
  return svg;
}

function buildBoard(el, interactive) {
  el.innerHTML = "";
  el.appendChild(buildTraces());
  for (let i = 0; i < SIZE * SIZE; i++) {
    const b = document.createElement(interactive ? "button" : "div");
    b.className = "cell" + (CORES.has(i) ? " core" : "");
    b.dataset.i = i;
    if (interactive) {
      b.type = "button";
      b.setAttribute("role", "gridcell");
      b.setAttribute("aria-label", `${cellName(i)}${CORES.has(i) ? ", core, 3 points" : ""}, empty`);
      b.tabIndex = i === 24 ? 0 : -1;
    }
    el.appendChild(b);
  }
}

function setNode(cell, owner, anim, delay = 0) {
  let node = cell.querySelector(".node");
  if (!owner) { if (node) node.remove(); return; }
  if (!node) { node = document.createElement("div"); cell.appendChild(node); }
  node.className = `node ${owner > 0 ? "h" : "a"}`;
  node.style.animationDelay = "";
  if (anim) {
    void node.offsetWidth; // restart animation
    node.classList.add(anim);
    node.style.animationDelay = `${delay}ms`;
  }
}

/* ---------------- demo board on the start screen ---------------- */
function paintDemo() {
  const el = $("#demo-board");
  buildBoard(el, false);
  const h = [8, 9, 15, 16, 17, 24, 31, 0, 30, 37];
  const a = [10, 11, 18, 22, 23, 25, 32, 38, 39, 48, 26];
  const cells = el.querySelectorAll(".cell");
  h.forEach((i) => setNode(cells[i], 1));
  a.forEach((i) => setNode(cells[i], -1));
  if (!reducedMotion) {
    // a slow, single capture loop: one blue placement flips a red line
    let on = false;
    setInterval(() => {
      on = !on;
      setNode(cells[12], on ? 1 : 0, on ? "placed" : null);
      [11, 10].forEach((i, k) => setNode(cells[i], on ? 1 : -1, "flipped", on ? 220 + k * 90 : 0));
    }, 2600);
  }
}

/* ---------------- rendering ---------------- */
function render(game, last) {
  const prev = ui.game;
  ui.game = game;
  const boardEl = $("#board");
  const cells = boardEl.querySelectorAll(".cell");
  const humanTurn = game.to_move === "human" && !game.over;
  const flippedSet = new Set(last ? last.flipped : []);
  const origin = last ? last.cell : null;

  cells.forEach((cell, i) => {
    const v = game.board[i];
    const before = sameMatch(prev, game) ? prev.board[i] : 0;
    let anim = null, delay = 0;
    if (i === origin) anim = "placed";
    else if (flippedSet.has(i)) {
      anim = "flipped";
      const dr = Math.abs(Math.floor(i / SIZE) - Math.floor(origin / SIZE));
      const dc = Math.abs((i % SIZE) - (origin % SIZE));
      delay = 120 + 90 * Math.max(dr, dc);
    }
    if (v !== before || anim || !sameMatch(prev, game)) setNode(cell, v, anim, delay);
    if (i === origin && !reducedMotion) ripple(cell, v);

    const legal = humanTurn && v === 0;
    cell.classList.toggle("legal", legal);
    cell.setAttribute("aria-disabled", String(!legal));
    cell.classList.remove("flip-preview", "denied");
    cell.classList.toggle("last-move", i === origin);
    cell.classList.toggle("by-ai", i === origin && last.by === "ai");
    cell.classList.toggle("by-human", i === origin && last.by === "human");
    const owner = v > 0 ? "you" : v < 0 ? "AI" : "empty";
    cell.setAttribute("aria-label", `${cellName(i)}${CORES.has(i) ? ", core, 3 points" : ""}, ${owner}`);
  });
  clearPreview();
  boardEl.classList.toggle("locked", !humanTurn);

  // scores
  $("#score-human").textContent = game.scores.human;
  $("#score-ai").textContent = game.scores.ai;
  $("#bar-human").style.flexGrow = Math.max(game.scores.human, 0.001);
  $("#bar-ai").style.flexGrow = Math.max(game.scores.ai, 0.001);
  $("#score-human-wrap").classList.toggle("active", game.to_move === "human");
  $("#score-ai-wrap").classList.toggle("active", game.to_move === "ai");

  const turn = $("#turn");
  turn.className = "turn " + (game.over ? "" : game.to_move);
  turn.textContent = game.over
    ? (game.winner === "human" ? "You won the match" : game.winner === "ai" ? "The AI won the match" : "Draw")
    : game.to_move === "human" ? "Your move" : "AI is thinking…";
  $("#ply").textContent = `Move ${game.ply} of 49. First to 30 points wins.`;

  renderLog(game.events);
  renderGauge(game.ai_win_prob);
  renderTrace(game.prob_trace);
  if (last && last.by === "ai") ui.lastAi = last;
  if (!sameMatch(prev, game)) ui.lastAi = null;
  renderDecision();
  renderRatings();
}

function sameMatch(a, b) {
  return Boolean(a && b && a.match.id === b.match.id);
}

function ripple(cell, v) {
  const r = document.createElement("span");
  r.className = `ripple ${v > 0 ? "h" : "a"}`;
  cell.appendChild(r);
  setTimeout(() => r.remove(), 750);
}

function renderLog(events) {
  const log = $("#log");
  log.innerHTML = "";
  [...events].reverse().forEach((e) => {
    const li = document.createElement("li");
    const gained = cellValue(e.cell) + e.flipped.reduce((s, j) => s + cellValue(j), 0);
    const caps = e.flipped.length ? `captured ${e.flipped.length}` : "placed";
    li.innerHTML = `<span class="n">${e.ply}</span>
      <span><span class="who-${e.by}">${e.by === "human" ? "You" : "AI"}</span> ${cellName(e.cell)}, ${caps}</span>
      <span class="gain">+${gained}</span>`;
    log.appendChild(li);
  });
  if (!events.length) log.innerHTML = '<li class="muted"><span></span><span>No moves yet</span><span></span></li>';
}

function renderGauge(p) {
  $("#ai-prob").textContent = pct(p);
  $("#gauge-fill").style.width = `${p * 100}%`;
}

function renderTrace(trace) {
  const svg = $("#trace");
  const W = 300, H = 110, pad = 6;
  const x = (k) => (k / 49) * W;
  const y = (p) => pad + (1 - p) * (H - 2 * pad);
  let html = `<line class="mid" x1="0" x2="${W}" y1="${y(0.5)}" y2="${y(0.5)}"/>`;
  if (trace.length > 1) {
    const pts = trace.map((p, k) => `${x(k).toFixed(1)},${y(p).toFixed(1)}`);
    html += `<path class="area" d="M0,${y(0.5)} L${pts.join(" L")} L${x(trace.length - 1)},${y(0.5)} Z"/>`;
    html += `<polyline class="line" points="${pts.join(" ")}"/>`;
  }
  html += `<line class="cursor" id="trace-cursor" x1="0" x2="0" y1="0" y2="${H}" visibility="hidden"/>`;
  svg.innerHTML = html;
}

function bindTraceHover() {
  const svg = $("#trace"), tip = $("#trace-tip");
  svg.addEventListener("mousemove", (ev) => {
    const trace = ui.game ? ui.game.prob_trace : [];
    if (!trace.length) return;
    const rect = svg.getBoundingClientRect();
    const k = Math.max(0, Math.min(trace.length - 1, Math.round(((ev.clientX - rect.left) / rect.width) * 49)));
    const cursor = $("#trace-cursor");
    const xPos = (k / 49) * 300;
    cursor.setAttribute("x1", xPos); cursor.setAttribute("x2", xPos); cursor.setAttribute("visibility", "visible");
    tip.hidden = false;
    tip.textContent = `${k === 0 ? "Start" : `After move ${k}`}: AI ${pct(trace[k])}%`;
    const left = Math.min(rect.width - tip.offsetWidth, Math.max(0, (k / 49) * rect.width - tip.offsetWidth / 2));
    tip.style.left = `${left}px`;
  });
  svg.addEventListener("mouseleave", () => {
    tip.hidden = true;
    const c = $("#trace-cursor"); if (c) c.setAttribute("visibility", "hidden");
  });
}

function renderDecision() {
  const box = $("#decision");
  const d = ui.lastAi;
  if (!d) { box.innerHTML = '<p class="muted">The AI hasn\'t moved yet.</p>'; return; }
  const top = d.candidates.slice(0, 5);
  const max = Math.max(...top.map((c) => c.p), 0.01);
  const total = (d.all_candidates || d.candidates).length;
  box.innerHTML = `
    <p>Played <b>${cellName(d.cell)}</b>. Its predicted win chance after that move: <b>${pct(d.chosen_p)}%</b>.</p>
    <ul class="cands">${top.map((c) => `
      <li class="${c.cell === d.cell ? "chosen" : ""}">
        <span>${cellName(c.cell)}</span>
        <span class="bar"><span style="width:${(c.p / max) * 100}%"></span></span>
        <span>${pct(c.p)}%</span>
      </li>`).join("")}</ul>
    <p class="meta">Rated ${total} possible move${total === 1 ? "" : "s"} in ${Math.round(d.think_ms)} ms (${d.method}).</p>`;
}

function renderRatings() {
  $$("#board .rating").forEach((r) => r.remove());
  const d = ui.lastAi, g = ui.game;
  if (!$("#toggle-thinking").checked || !d || !d.all_candidates || !g || g.over || g.to_move !== "human") return;
  const cells = $("#board").querySelectorAll(".cell");
  const ps = d.all_candidates.map((c) => c.p);
  const lo = Math.min(...ps), hi = Math.max(...ps);
  d.all_candidates.forEach((c, rank) => {
    if (g.board[c.cell] !== 0) return; // the chosen cell (and any captured) is no longer empty
    const t = hi > lo ? (c.p - lo) / (hi - lo) : 1;
    const el = document.createElement("span");
    el.className = "rating" + (rank < 3 ? " top" : "");
    el.style.background = `rgba(229, 72, 77, ${0.18 + 0.72 * t})`;
    el.textContent = pct(c.p);
    el.title = `The AI rated ${cellName(c.cell)} at ${pct(c.p)}% for itself`;
    cells[c.cell].appendChild(el);
  });
}

/* ---------------- hover preview ---------------- */
function showPreview(i) {
  const g = ui.game;
  if (ui.previewCell === i) return; // events from the ghost itself must not rebuild it
  clearPreview();
  if (!g || ui.busy || g.over || g.to_move !== "human" || g.board[i] !== 0) return;
  ui.previewCell = i;
  const cells = $("#board").querySelectorAll(".cell");
  const flips = g.preview[String(i)] || [];
  const ghost = document.createElement("div");
  ghost.className = "node h ghost";
  cells[i].appendChild(ghost);
  flips.forEach((j) => cells[j].classList.add("flip-preview"));
  const gain = cellValue(i) + flips.reduce((s, j) => s + cellValue(j), 0);
  const tag = document.createElement("span");
  tag.className = "gain-tag";
  tag.textContent = `+${gain}`;
  cells[i].appendChild(tag);
}

function clearPreview() {
  ui.previewCell = null;
  $$("#board .ghost, #board .gain-tag").forEach((n) => n.remove());
  $$("#board .flip-preview").forEach((n) => n.classList.remove("flip-preview"));
}

/* ---------------- game flow ---------------- */
function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), ms);
}

async function startGame() {
  if (ui.busy) return;
  ui.busy = true;
  try {
    const game = await api("/api/new_game", ui.settings);
    $("#end").hidden = true;
    showScreen("game");
    ui.game = null;
    ui.lastAi = null;
    buildBoard($("#board"), true);
    $("#chip-diff").textContent = (ui.status?.difficulties?.[game.difficulty]?.label) || game.difficulty;
    render(game, null);
    ui.busy = false;
    if (game.to_move === "ai") await aiTurn();
    else focusCell(24);
  } catch (e) {
    ui.busy = false;
    showScreen("start");
    banner(escapeHtml(e.message));
  }
}

async function humanMove(i) {
  const g = ui.game;
  if (!g || g.over) return;
  const cell = $("#board").querySelectorAll(".cell")[i];
  if (ui.busy || g.to_move !== "human") { toast("Wait for the AI to finish its move."); return; }
  if (g.board[i] !== 0) {
    cell.classList.remove("denied"); void cell.offsetWidth; cell.classList.add("denied");
    toast(`${cellName(i)} is already taken. Pick an empty cell.`);
    return;
  }
  ui.busy = true;
  try {
    const next = await api("/api/move", { match: g.match, cell: i });
    render(next, next.last);
    if (next.last.flipped.length) toast(`You captured ${next.last.flipped.length} node${next.last.flipped.length > 1 ? "s" : ""}.`);
    if (next.over) { await sleep(900); return showEnd(next); }
    await sleep(reducedMotion ? 150 : 550);
    ui.busy = false;
    await aiTurn();
  } catch (e) {
    toast(e.message);
  } finally {
    ui.busy = false;
  }
}

async function aiTurn() {
  const g = ui.game;
  if (!g || g.over || g.to_move !== "ai") return;
  ui.busy = true;
  $("#turn").textContent = `AI is rating ${49 - g.ply} possible moves…`;
  try {
    const [next] = await Promise.all([api("/api/ai_move", { match: g.match }), sleep(reducedMotion ? 0 : 450)]);
    render(next, next.last);
    const n = next.last.flipped.length;
    if (n) toast(`The AI captured ${n} of your nodes.`);
    if (next.over) { await sleep(900); showEnd(next); }
  } catch (e) {
    toast(e.message);
  } finally {
    ui.busy = false;
  }
}

function showEnd(g) {
  const won = g.winner === "human";
  const title = $("#end-title");
  title.textContent = won ? "You win" : g.winner === "ai" ? "The AI wins" : "Draw";
  title.className = "end-title " + (won ? "win" : "loss");
  $("#end-kicker").textContent = `${ui.status?.difficulties?.[g.difficulty]?.label || g.difficulty} match over`;
  $("#end-score").textContent = `${g.scores.human} – ${g.scores.ai}`;
  const tr = g.prob_trace.slice(1, -1);
  let note = "";
  if (tr.length) {
    if (won) {
      const k = tr.indexOf(Math.max(...tr));
      note = `At its most confident, after move ${k + 1}, the model gave itself ${pct(tr[k])}% to win. You turned it around.`;
    } else {
      const k = tr.indexOf(Math.min(...tr));
      note = `The model's lowest estimate for itself was ${pct(tr[k])}%, after move ${k + 1}.`;
    }
  }
  saveResult(g);
  $("#end-note").textContent = note + " This match was saved to your history.";
  $("#end").hidden = false;
  $("#btn-rematch").focus();
}

function focusCell(i) {
  const cells = $("#board").querySelectorAll(".cell");
  cells.forEach((c) => (c.tabIndex = -1));
  if (cells[i]) { cells[i].tabIndex = 0; cells[i].focus({ preventScroll: true }); }
}

/* ---------------- screens & dialogs ---------------- */
function showScreen(name) {
  $("#screen-start").hidden = name !== "start";
  $("#screen-game").hidden = name !== "game";
  if (name === "start") $("#end").hidden = true;
}

function banner(msg) {
  const b = $("#model-banner");
  if (!msg) { b.hidden = true; return; }
  b.hidden = false;
  b.innerHTML = msg;
}

async function loadStatus() {
  try {
    ui.status = await api("/api/status");
  } catch (e) {
    banner(escapeHtml(e.message));
    return;
  }
  const s = ui.status;
  if (!s.model_loaded) {
    banner(`<b>The trained model is missing.</b> ${escapeHtml(s.error || "")}<br>
      From the project folder, run <code>python scripts/generate_dataset.py</code> and then
      <code>python scripts/train_model.py</code>. Restart the game when they finish.`);
    $("#btn-start").disabled = true;
  } else if (s.model && s.model.train_rows) {
    $("#lede-rows").textContent = s.model.train_rows.toLocaleString();
    $("#lede-games").textContent = s.model.train_games.toLocaleString();
  }
}

function escapeHtml(t) {
  return String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* Match history lives in this browser (localStorage), so it also works on a
   serverless host. Running locally, the server additionally logs finished
   matches to data/player_games/matches.jsonl for retraining. */
const HISTORY_KEY = "circuit-siege-history";

function loadHistory() {
  try {
    const h = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(h) ? h.filter((r) => r && ["human", "ai", "draw"].includes(r.winner)) : [];
  } catch {
    return [];
  }
}

function localTimestamp() {
  const d = new Date(), p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function saveResult(g) {
  const hist = loadHistory();
  if (hist.some((r) => r.id === g.match.id)) return;
  hist.push({
    id: g.match.id, finished_at: localTimestamp(), difficulty: g.difficulty,
    score_human: g.scores.human, score_ai: g.scores.ai, winner: g.winner,
  });
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(hist.slice(-200))); } catch { /* storage full or blocked */ }
}

function historyStats(hist) {
  const by = {};
  for (const d of ["easy", "normal", "hard"]) {
    const rs = hist.filter((r) => r.difficulty === d);
    by[d] = { played: rs.length, human_wins: rs.filter((r) => r.winner === "human").length };
  }
  let streak = 0;
  for (let k = hist.length - 1; k >= 0 && hist[k].winner === "human"; k--) streak++;
  return {
    played: hist.length,
    human_wins: hist.filter((r) => r.winner === "human").length,
    ai_wins: hist.filter((r) => r.winner === "ai").length,
    by_difficulty: by, win_streak: streak, recent: hist.slice(-10).reverse(),
  };
}

function fillStats() {
  const body = $("#stats-body");
  try {
    const s = historyStats(loadHistory());
    if (!s.played) { body.innerHTML = '<p class="muted">No finished matches yet. Your results appear here after your first game.</p>'; return; }
    const labels = ui.status?.difficulties || {};
    const rows = ["easy", "normal", "hard"].filter((d) => s.by_difficulty[d]).map((d) => [d, s.by_difficulty[d]]).map(([d, v]) =>
      `<tr><td>${escapeHtml(labels[d]?.label || d)}</td><td>${v.played}</td><td>${v.human_wins}</td><td>${v.played ? pct(v.human_wins / v.played) + "%" : "–"}</td></tr>`).join("");
    const recent = s.recent.map((r) =>
      `<tr><td>${escapeHtml(String(r.finished_at).replace("T", " ").slice(0, 16))}</td><td>${escapeHtml(labels[r.difficulty]?.label || r.difficulty)}</td>
       <td>${r.score_human}–${r.score_ai}</td><td class="${r.winner === "human" ? "res-win" : "res-loss"}">${r.winner === "human" ? "Won" : r.winner === "ai" ? "Lost" : "Draw"}</td></tr>`).join("");
    body.innerHTML = `
      <div class="stat-grid">
        <div class="stat"><b>${s.played}</b><span>matches played</span></div>
        <div class="stat"><b>${s.human_wins}–${s.ai_wins}</b><span>your wins – AI wins</span></div>
        <div class="stat"><b>${s.win_streak}</b><span>current win streak</span></div>
      </div>
      <table class="tbl"><thead><tr><th>Level</th><th>Played</th><th>You won</th><th>Win rate</th></tr></thead><tbody>${rows}</tbody></table>
      <h3>Recent matches</h3>
      <table class="tbl"><thead><tr><th>When</th><th>Level</th><th>Score</th><th>Result</th></tr></thead><tbody>${recent}</tbody></table>`;
  } catch (e) {
    body.innerHTML = `<p class="muted">${escapeHtml(e.message)}</p>`;
  }
}

function fillModel() {
  const body = $("#model-body");
  const s = ui.status;
  if (!s || !s.model_loaded) { body.innerHTML = '<p class="muted">No trained model is loaded.</p>'; return; }
  const m = s.model, t = m.test_metrics || {};
  const fmt = (v, p = false) => (v === undefined || v === null ? "–" : p ? `${(v * 100).toFixed(1)}%` : v.toFixed(3));
  const LEVEL = { easy: "Rookie", normal: "Operator", hard: "Architect" };
  const ORDER = ["easy", "normal", "hard", "random", "greedy", "cautious", "positional"];
  const rank = (k) => k.split(" vs ").map((x) => ORDER.indexOf(x)).reduce((a, x) => a * 10 + x, 0);
  const label = (k) => k.split(" vs ").map((x, i) => LEVEL[x] || (i ? `${x} bot` : x)).join(" vs ");
  const arena = m.arena ? Object.entries(m.arena).sort(([a], [b]) => rank(a) - rank(b)).map(([k, v]) =>
    `<tr><td>${escapeHtml(label(k))}</td><td>${fmt(v.win_rate, true)}</td><td>${fmt(v.ci95[0], true)} to ${fmt(v.ci95[1], true)}</td></tr>`).join("") : "";
  body.innerHTML = `
    <p>The AI never follows a hand-written strategy. On each turn it imagines every legal move and turns each
       resulting board into ${68} numbers, such as cells owned, cores held, and how much each side can capture next.
       A trained classifier then predicts the AI's chance of winning from each board. The AI picks
       from the best-rated moves; the level decides how strictly.</p>
    <div class="pipeline">
      <div><b>Simulate</b>Scripted bots play thousands of varied games.</div>
      <div><b>Label</b>Every position is tagged with who eventually won.</div>
      <div><b>Train</b>Four model types compared; the best is kept.</div>
      <div><b>Play</b>The model rates every candidate move during your match.</div>
    </div>
    <h3>Loaded model</h3>
    <dl class="kv">
      <dt>Algorithm</dt><dd>${escapeHtml(m.algorithm.replaceAll("_", " "))}</dd>
      <dt>Version</dt><dd>${escapeHtml(m.version)} (trained ${escapeHtml(m.trained_at.replace("T", " "))})</dd>
      <dt>Training data</dt><dd>${m.train_rows.toLocaleString()} positions from ${m.train_games.toLocaleString()} games (${(m.data_sources || []).join(", ")})</dd>
      <dt>Test accuracy</dt><dd>${fmt(t.accuracy, true)} on games it never saw</dd>
      <dt>Test ROC-AUC</dt><dd>${fmt(t.roc_auc)}</dd>
      <dt>Test log-loss</dt><dd>${fmt(t.log_loss)} (a coin flip scores 0.693; lower is better)</dd>
    </dl>
    ${arena ? `<h3>Head-to-head against scripted bots</h3>
    <table class="tbl"><thead><tr><th>Matchup</th><th>AI win rate</th><th>95% interval</th></tr></thead><tbody>${arena}</tbody></table>` :
    '<p class="hint">Run <code>python scripts/evaluate_model.py</code> to add test and head-to-head results here.</p>'}
    <p class="hint">The full write-up, charts included, is in <code>reports/evaluation_report.md</code>.</p>`;
}

/* ---------------- wiring ---------------- */
function bind() {
  $$('input[name="difficulty"]').forEach((r) => r.addEventListener("change", () => (ui.settings.difficulty = r.value)));
  $$('input[name="first"]').forEach((r) => r.addEventListener("change", () => (ui.settings.first = r.value)));
  $("#btn-start").addEventListener("click", startGame);
  $("#btn-restart").addEventListener("click", startGame);
  $("#btn-rematch").addEventListener("click", startGame);
  $("#btn-review").addEventListener("click", () => { $("#end").hidden = true; });
  $("#btn-end-menu").addEventListener("click", () => showScreen("start"));
  $("#btn-menu").addEventListener("click", () => showScreen("start"));
  $("#toggle-thinking").addEventListener("change", renderRatings);

  $$("[data-open]").forEach((b) => b.addEventListener("click", () => {
    const dlg = document.getElementById(b.dataset.open);
    if (dlg.id === "dlg-stats") fillStats();
    if (dlg.id === "dlg-model") fillModel();
    dlg.showModal();
  }));

  const board = $("#board");
  board.addEventListener("click", (e) => {
    const cell = e.target.closest(".cell");
    if (cell) humanMove(Number(cell.dataset.i));
  });
  board.addEventListener("mouseover", (e) => {
    const cell = e.target.closest(".cell");
    if (cell) showPreview(Number(cell.dataset.i));
  });
  board.addEventListener("mouseleave", clearPreview);
  board.addEventListener("focusin", (e) => {
    const cell = e.target.closest(".cell");
    if (cell) showPreview(Number(cell.dataset.i));
  });
  board.addEventListener("keydown", (e) => {
    const cell = e.target.closest(".cell");
    if (!cell) return;
    const i = Number(cell.dataset.i), r = Math.floor(i / SIZE), c = i % SIZE;
    const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    if (moves[e.key]) {
      e.preventDefault();
      const nr = Math.min(SIZE - 1, Math.max(0, r + moves[e.key][0]));
      const nc = Math.min(SIZE - 1, Math.max(0, c + moves[e.key][1]));
      focusCell(nr * SIZE + nc);
    }
  });
  bindTraceHover();
}

bind();
paintDemo();
buildBoard($("#board"), true);
loadStatus();
