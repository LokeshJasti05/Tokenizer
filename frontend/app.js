/**
 * app.js — Tokenizer frontend (simplified single-prompt mode)
 *
 * State: one string `promptText`
 * Flow: user types → build raw prompt → send to /api/tokenize → render results
 */

const API_BASE = '';   // same origin

// ─── State ───────────────────────────────────────────────────────────────────

let promptText     = '';
let currentTokenIds = [];
let showWs         = false;
let debounceTimer  = null;
const DEBOUNCE_MS  = 200;

// ─── DOM refs ─────────────────────────────────────────────────────────────────

const promptInput         = document.getElementById('promptInput');
const rawPromptArea       = document.getElementById('rawPromptArea');
const tokenCountValue     = document.getElementById('tokenCountValue');
const tokenVis            = document.getElementById('tokenVis');
const tokenIdsOutput      = document.getElementById('tokenIdsOutput');
const loadingBar          = document.getElementById('loadingBar');
const showWhitespaceToggle= document.getElementById('showWhitespace');
const copyIdsBtn          = document.getElementById('copyIdsBtn');
const copyToast           = document.getElementById('copyToast');
const historyList         = document.getElementById('historyList');
const refreshHistoryBtn   = document.getElementById('refreshHistoryBtn');

// ─── Build raw prompt ──────────────────────────────────────────────────────
// Pass the user's text through directly — no special token wrapping

function buildRawPrompt(text) {
  return text;
}

// ─── Input listener ────────────────────────────────────────────────────────

promptInput.addEventListener('input', () => {
  promptText = promptInput.value;

  // Immediately update raw prompt display
  const raw = buildRawPrompt(promptText);
  rawPromptArea.value = raw;

  // Debounce the API call
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => doTokenize(raw), DEBOUNCE_MS);
});

// ─── Tokenise via API ──────────────────────────────────────────────────────

async function doTokenize(rawText) {
  if (!rawText) {
    updateTokenCount(0);
    renderTokenVis([]);
    renderTokenIds([]);
    return;
  }

  loadingBar.classList.add('active');

  try {
    const model = document.getElementById('modelSelect').value;
    const resp = await fetch(`${API_BASE}/api/tokenize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: rawText, model }),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    currentTokenIds = data.token_ids || [];
    updateTokenCount(data.token_count || 0);
    renderTokenVis(data.segments || []);
    renderTokenIds(data.token_ids || []);

    // Auto-save to history (2-second settle, only when non-empty)
    if (rawText.trim()) autoSave(rawText, data);

  } catch (err) {
    console.error('Tokenize error:', err);
  } finally {
    loadingBar.classList.remove('active');
  }
}

// ─── Token count animation ─────────────────────────────────────────────────

let lastCount = 0;
function updateTokenCount(count) {
  if (count === lastCount) return;
  lastCount = count;
  tokenCountValue.textContent = count.toLocaleString();
  tokenCountValue.classList.add('bump');
  setTimeout(() => tokenCountValue.classList.remove('bump'), 350);
}

// ─── Token visualization ───────────────────────────────────────────────────

const NUM_COLORS = 10;

function renderTokenVis(segments) {
  if (!segments || segments.length === 0) {
    tokenVis.innerHTML = '<span class="placeholder-text">Start typing to see token colors…</span>';
    return;
  }

  tokenVis.innerHTML = '';

  segments.forEach((seg, i) => {
    const chip = document.createElement('span');
    chip.className = `token-chip tok-${i % NUM_COLORS}`;
    chip.dataset.id = seg.id;
    chip.title = `Token ID: ${seg.id}`;
    chip.setAttribute('role', 'mark');
    chip.setAttribute('aria-label', `Token ${seg.id}: "${seg.text}"`);

    let display = seg.text;
    if (showWs) {
      display = display.replace(/ /g, '·').replace(/\n/g, '↵\n').replace(/\t/g, '→\t');
    }
    chip.textContent = display || ' ';
    tokenVis.appendChild(chip);
  });
}

// ─── Token IDs ─────────────────────────────────────────────────────────────

function renderTokenIds(ids) {
  if (!ids || ids.length === 0) {
    tokenIdsOutput.innerHTML = '<span class="placeholder-text">—</span>';
    return;
  }
  tokenIdsOutput.textContent = ids.join(', ');
}

// ─── Copy ──────────────────────────────────────────────────────────────────

copyIdsBtn.addEventListener('click', () => {
  if (!currentTokenIds.length) return;
  navigator.clipboard.writeText(currentTokenIds.join(', ')).then(() => showToast('Copied to clipboard!'));
});

function showToast(msg) {
  copyToast.textContent = msg;
  copyToast.classList.add('show');
  setTimeout(() => copyToast.classList.remove('show'), 2200);
}

// ─── Whitespace toggle ─────────────────────────────────────────────────────

showWhitespaceToggle.addEventListener('change', () => {
  showWs = showWhitespaceToggle.checked;
  const raw = buildRawPrompt(promptText);
  doTokenize(raw);
});

// ─── Model selector ────────────────────────────────────────────────────────

document.getElementById('modelSelect').addEventListener('change', () => {
  const raw = buildRawPrompt(promptText);
  doTokenize(raw);
});

// ─── History ───────────────────────────────────────────────────────────────

let _autoSaveTimer = null;
function autoSave(rawText, data) {
  clearTimeout(_autoSaveTimer);
  _autoSaveTimer = setTimeout(async () => {
    try {
      await fetch(`${API_BASE}/api/history`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages:    [{ role: 'user', content: promptText }],
          token_count: data.token_count,
          token_ids:   data.token_ids,
          raw_text:    rawText,
          model:       document.getElementById('modelSelect').value,
        }),
      });
      loadHistory();
    } catch (_) { /* history is non-critical */ }
  }, 2000);
}

async function loadHistory() {
  try {
    const resp = await fetch(`${API_BASE}/api/history`);
    if (!resp.ok) return;
    const data = await resp.json();
    renderHistory(data.history || []);
  } catch (_) {}
}

function renderHistory(items) {
  if (!items.length) {
    historyList.innerHTML = '<span class="placeholder-text">No sessions yet</span>';
    return;
  }

  historyList.innerHTML = '';
  items.slice(0, 20).forEach(item => {
    const el = document.createElement('div');
    el.className = 'history-item';
    el.setAttribute('tabindex', '0');
    el.setAttribute('role', 'button');
    el.setAttribute('aria-label', `Session: ${item.token_count} tokens`);

    const msgs = item.messages || [];
    const preview = msgs.length ? (msgs[0].content || '') : (item.raw_text || '');

    const textEl = document.createElement('div');
    textEl.className = 'history-item-text';
    textEl.textContent = preview.slice(0, 60) || '(empty)';

    const metaEl = document.createElement('div');
    metaEl.className = 'history-item-meta';
    const d = new Date(item.created_at + 'Z');
    metaEl.textContent = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const badge = document.createElement('span');
    badge.className = 'history-badge';
    badge.textContent = item.token_count;

    el.appendChild(textEl);
    el.appendChild(metaEl);
    el.appendChild(badge);

    // Restore on click
    el.addEventListener('click', () => restoreSession(item));
    el.addEventListener('keydown', e => { if (e.key === 'Enter') restoreSession(item); });

    historyList.appendChild(el);
  });
}

function restoreSession(item) {
  const msgs = item.messages || [];
  const text = msgs.length ? (msgs[0].content || '') : '';
  promptInput.value = text;
  promptText = text;
  const raw = buildRawPrompt(text);
  rawPromptArea.value = raw;
  doTokenize(raw);
}

refreshHistoryBtn.addEventListener('click', loadHistory);

// ─── Init ──────────────────────────────────────────────────────────────────

(function init() {
  promptInput.focus();
  loadHistory();
})();
