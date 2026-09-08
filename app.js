// HF Model Explorer — client-side search/sort/filter over the Hugging Face Hub API.
// No build step. The browser talks to https://huggingface.co/api/models directly (CORS is open).

const API = 'https://huggingface.co/api/models';
const EXPAND = ['downloads', 'downloadsAllTime', 'likes', 'lastModified', 'createdAt', 'library_name',
  'pipeline_tag', 'tags', 'safetensors', 'gated', 'trendingScore', 'author', 'gguf', 'baseModels',
  'inferenceProviderMapping', 'siblings'];
const PAGE = 1000;
const SEP = '';

const PIPELINES = ['text-generation', 'image-text-to-text', 'text2text-generation', 'feature-extraction', 'sentence-similarity',
  'text-classification', 'token-classification', 'question-answering', 'summarization', 'translation', 'fill-mask',
  'zero-shot-classification', 'text-ranking', 'text-to-image', 'image-to-image', 'image-to-text', 'text-to-video',
  'image-to-video', 'text-to-speech', 'text-to-audio', 'automatic-speech-recognition', 'audio-classification',
  'audio-text-to-text', 'image-classification', 'object-detection', 'image-segmentation', 'depth-estimation',
  'zero-shot-image-classification', 'video-classification', 'visual-question-answering', 'document-question-answering',
  'any-to-any', 'reinforcement-learning', 'robotics', 'tabular-classification', 'tabular-regression', 'time-series-forecasting',
  'image-feature-extraction', 'mask-generation', 'visual-document-retrieval', 'keypoint-detection', 'video-text-to-text'];
const LIBRARIES = ['transformers', 'diffusers', 'gguf', 'mlx', 'peft', 'sentence-transformers', 'safetensors', 'pytorch',
  'onnx', 'transformers.js', 'timm', 'vllm', 'llama.cpp', 'stable-baselines3', 'ultralytics', 'keras', 'tensorflow', 'jax',
  'adapter-transformers', 'setfit', 'spacy', 'fasttext', 'open_clip', 'nemo', 'speechbrain', 'espnet', 'paddlenlp', 'mlc-llm'];

// Plain tags that look like ISO language codes but are not languages.
const NOT_LANG = new Set(['gguf', 'mlx', 'fp8', 'awq', 'trl', 'sft', 'dpo', 'grpo', 'ppo', 'rl', 'vlm', 'llm', 'vit', 'nlp', 'ocr',
  'asr', 'tts', 'api', 'cot', 'sql', 'rag', 'moe', 'ai', 'ml', 'cv', 'gan', 'vae', 'ner', 'qa', 'rlhf', 'bert', 'gpt', 'exl2',
  'exl3', 'hqq', 'bnb', 'int4', 'int8', 'nf4', 'lora', 'qlora', 'llava', 'clip', 'yolo', 'sam', 'sd', 'sdxl', 'flux', 'onnx',
  'tf', 'jax', 'code', 'chat', 'base', 'math', 'roleplay', 'rp', 'nsfw', 'sfw', 'orpo', 'kto', 'simpo', 'dare', 'ties', 'slerp']);
const LANG_RE = /^[a-z]{2,3}(?:[-_][A-Za-z]{2,4})?$/;
const QUANT_TAGS = new Set(['4-bit', '8-bit', '2-bit', '3-bit', '5-bit', '6-bit', 'gguf', 'awq', 'gptq', 'fp8', 'nvfp4', 'mxfp4',
  'bitsandbytes', 'compressed-tensors', 'exl2', 'exl3', 'hqq', 'int4', 'int8', 'quantized', 'quantization', 'llamacpp', 'imatrix',
  'autoround', 'auto-round', 'aqlm', 'eetq', 'quanto', 'torchao', 'onnx', 'openvino', 'coreml', 'tensorrt', 'nf4']);
const QUANT_NAME_RE = /(gguf|awq|gptq|exl[23]|hqq|bnb|nf4|fp8|nvfp4|mxfp4|w4a16|w8a8|w8a16|int[48]\b|[-_.](q[2-8]|iq[1-4]|[2-8]bit|[48]-?bit)\b|-(Q[2-8]_|IQ[1-4]_)|quantized|\bmlx\b|-\d+bit)/i;

// Parameter count parsed from the repo name: "27B", "30B-A3B" (total, not active), "8x7B", "360M", "E4B".
// Used when the Hub metadata is missing or clearly describes a different file (e.g. a draft/MTP GGUF).
const NAME_PARAMS_RE = /(?<![\d.])(?:(\d+)x)?(\d+(?:\.\d+)?)([bBmM])(?![a-z0-9])/g;
function paramsFromName(name) {
  let best = null;
  for (const m of name.matchAll(NAME_PARAMS_RE)) {
    const n = +m[2] * (m[1] ? +m[1] : 1) * (m[3].toLowerCase() === 'b' ? 1e9 : 1e6);
    if (n >= 1e6 && n <= 5e12 && (best == null || n > best)) best = n;
  }
  return best;
}

// GGUF files that are not the model weights: vision projectors, speculative-decoding draft heads, adapters.
const AUX_GGUF_RE = /mmproj|projector|draft|mtp|dflash|eagle|specul|medusa|-lora|adapter/i;
const QUANT_RE = /(IQ[1-4]_(?:XXS|XS|S|M|NL)|Q[2-8]_K(?:_[SMLPX]+)?|Q[2-8]_[01]|TQ[12]_0|BF16|F16|F32|MXFP4)/i;
// Approximate bits per weight of llama.cpp quant types, used to estimate parameter counts from file sizes.
const BPW = { F32: 32, BF16: 16, F16: 16, Q8_0: 8.5, Q6_K: 6.56, Q5_K_M: 5.69, Q5_K_S: 5.52, Q5_1: 6, Q5_0: 5.5, Q4_K_M: 4.85, Q4_K_S: 4.58,
  Q4_1: 5, Q4_0: 4.5, IQ4_NL: 4.5, IQ4_XS: 4.25, MXFP4: 4.25, Q3_K_L: 4.03, Q3_K_M: 3.91, Q3_K_S: 3.5, IQ3_M: 3.7, IQ3_S: 3.5, IQ3_XS: 3.3,
  IQ3_XXS: 3.1, Q2_K: 3.35, Q2_K_S: 3.0, IQ2_M: 2.7, IQ2_S: 2.5, IQ2_XS: 2.31, IQ2_XXS: 2.06, IQ1_M: 1.75, IQ1_S: 1.56, TQ2_0: 2.06, TQ1_0: 1.69 };
const BPW_FALLBACK = { Q8: 8.5, Q6: 6.6, Q5: 5.7, Q4: 4.8, Q3: 3.9, Q2: 3.2, IQ4: 4.4, IQ3: 3.5, IQ2: 2.5, IQ1: 1.7 };
const quantLabel = (path) => { const m = QUANT_RE.exec(path.split('/').pop()); return m ? m[1].toUpperCase() : null; };
const bpwFor = (label) => BPW[label] || BPW_FALLBACK[(/^(IQ\d|Q\d)/.exec(label) || [])[1]] || null;
const isAuxGguf = (path) => AUX_GGUF_RE.test(path);
const basename = (path) => path.split('/').pop();

const $ = (id) => document.getElementById(id);
const fmtInt = (n) => n == null ? '' : Math.round(n).toLocaleString('en-US');
const fmtCompact = (n) => n == null ? '' : n >= 1e9 ? (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M' : n >= 1e4 ? (n / 1e3).toFixed(0) + 'k' : fmtInt(n);
const fmtParams = (n) => n == null ? '' : n >= 1e9 ? (n / 1e9).toFixed(n >= 1e11 ? 0 : 1) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(0) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(0) + 'k' : String(n);
const fmtBytes = (n) => n == null ? '' : n >= 1e12 ? (n / 1e12).toFixed(2) + ' TB' : n >= 1e9 ? (n / 1e9).toFixed(1) + ' GB' : n >= 1e6 ? (n / 1e6).toFixed(0) + ' MB' : fmtInt(n) + ' B';
const fmtDate = (d) => d ? d.toISOString().slice(0, 10) : '';
const fmtPct = (x) => x == null ? '' : (x * 100).toFixed(x < 0.1 ? 1 : 0) + '%';
const fmtRatio = (x) => x == null ? '' : x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const daysAgo = (d, now) => d ? (now - d.getTime()) / 86400000 : null;

// ---------- Column definitions ----------
const COLUMNS = [
  { key: 'id', label: 'model', get: (r) => r.id, type: 'str', on: true, cls: 'id' },
  { key: 'author', label: 'author', get: (r) => r.author, type: 'str', on: false },
  { key: 'params', label: 'params', get: (r) => r.params, type: 'num', fmt: fmtParams, on: true, title: 'Parameter count from safetensors metadata, else GGUF metadata; ~ marks a value parsed from the model name (metadata missing or describing a different file)' },
  { key: 'downloads', label: 'dl 30d', get: (r) => r.downloads, type: 'num', fmt: fmtCompact, on: true },
  { key: 'downloadsAllTime', label: 'dl total', get: (r) => r.downloadsAllTime, type: 'num', fmt: fmtCompact, on: true },
  { key: 'likes', label: 'likes', get: (r) => r.likes, type: 'num', fmt: fmtCompact, on: true },
  { key: 'trending', label: 'trend', get: (r) => r.trending, type: 'num', fmt: fmtInt, on: true, title: 'Hub trending score' },
  { key: 'momentum', label: 'momentum', get: (r) => r.momentum, type: 'num', fmt: fmtPct, on: true, title: '30-day downloads as a share of all-time downloads. High = recent surge or brand new.' },
  { key: 'dlPerDay', label: 'dl/day', get: (r) => r.dlPerDay, type: 'num', fmt: fmtCompact, on: false, title: 'All-time downloads divided by days since creation' },
  { key: 'likesPerK', label: 'likes/10k dl', get: (r) => r.likesPerK, type: 'num', fmt: fmtRatio, on: false, title: 'Likes per 10,000 all-time downloads (quality-ish signal)' },
  { key: 'created', label: 'created', get: (r) => r.created, type: 'date', fmt: fmtDate, on: true },
  { key: 'modified', label: 'modified', get: (r) => r.modified, type: 'date', fmt: fmtDate, on: true },
  { key: 'ageDays', label: 'age (d)', get: (r) => r.ageDays, type: 'num', fmt: (x) => x == null ? '' : Math.floor(x), on: false },
  { key: 'arch', label: 'arch', get: (r) => r.arch, type: 'str', on: true },
  { key: 'pipeline', label: 'task', get: (r) => r.pipeline, type: 'str', on: true },
  { key: 'library', label: 'library', get: (r) => r.library, type: 'str', on: false },
  { key: 'license', label: 'license', get: (r) => r.license, type: 'str', on: true },
  { key: 'quant', label: 'quant', get: (r) => r.quant, type: 'str', on: false, title: 'Detected quantization format' },
  { key: 'ctx', label: 'gguf ctx', get: (r) => r.ctx, type: 'num', fmt: fmtCompact, on: false, title: 'Context length from GGUF metadata' },
  { key: 'ggufSize', label: 'gguf size', get: (r) => r.ggufSize, type: 'num', fmt: fmtBytes, on: false, title: 'Size of the smallest main GGUF quant (after check files), else the single file the Hub reports' },
  { key: 'quants', label: 'gguf quants', get: (r) => r.quants.join(' '), type: 'str', on: false, title: 'Quant variants of the main model weights present in the repo' },
  { key: 'repoBytes', label: 'repo size', get: (r) => r.repoBytes, type: 'num', fmt: fmtBytes, on: false, title: 'Total size of all files (after check files)' },
  { key: 'providers', label: 'providers', get: (r) => r.providersLive, type: 'num', fmt: fmtInt, on: false, title: 'Number of live inference providers' },
  { key: 'base', label: 'base model', get: (r) => r.baseIds[0] || '', type: 'str', on: false },
  { key: 'relation', label: 'relation', get: (r) => r.relation, type: 'str', on: false, title: 'How this repo relates to its base model' },
  { key: 'langs', label: 'langs', get: (r) => r.langs.length, type: 'num', fmt: fmtInt, on: false, title: 'Number of language tags' },
];
const COL = Object.fromEntries(COLUMNS.map((c) => [c.key, c]));

// ---------- State ----------
const state = {
  rows: [],          // normalized rows from last fetch
  view: [],          // filtered + sorted
  sort: [{ key: 'downloads', dir: -1 }],
  page: 0,
  pageSize: 100,
  cols: new Set(COLUMNS.filter((c) => c.on).map((c) => c.key)),
  facetSel: {},      // facetKey -> Set of selected values
  open: new Set(),   // expanded row ids
  fetchedWith: '',   // fetch params signature of current rows
  abort: null,
  fcAbort: null,     // AbortController of a running batch file check
};

// ---------- Normalization ----------
function normalize(m, now) {
  const tags = Array.isArray(m.tags) ? m.tags : [];
  const created = m.createdAt ? new Date(m.createdAt) : null;
  const modified = m.lastModified ? new Date(m.lastModified) : null;
  const st = m.safetensors && m.safetensors.total;
  const gg = m.gguf;
  const fromName = paramsFromName(m.id.split('/').pop() || '');
  let params = st || (gg && gg.total) || null;
  let paramsSrc = st ? 'safetensors' : params ? 'gguf' : null;
  const license = tags.filter((t) => t.startsWith('license:')).map((t) => t.slice(8)).join(', ') || null;
  const langs = tags.filter((t) => !t.includes(':') && LANG_RE.test(t) && !NOT_LANG.has(t));
  const datasets = tags.filter((t) => t.startsWith('dataset:')).map((t) => t.slice(8));
  const arxiv = tags.filter((t) => t.startsWith('arxiv:')).map((t) => t.slice(6));
  const cfg = m.config || {};
  const arch = cfg.model_type || (gg && gg.architecture) || null;
  const archs = Array.isArray(cfg.architectures) ? cfg.architectures : [];
  const rel = {};
  const baseIds = [];
  // baseModels is a single object when there is one relation, an array when there are several.
  const bms = Array.isArray(m.baseModels) ? m.baseModels : m.baseModels ? [m.baseModels] : [];
  for (const b of bms) {
    const ids = (Array.isArray(b.models) ? b.models : []).map((x) => x.id).filter(Boolean);
    if (!b.relation || !ids.length) continue;
    rel[b.relation] = (rel[b.relation] || []).concat(ids);
    baseIds.push(...ids);
  }
  const relation = Object.keys(rel).sort().join('+') || null;
  const tagSet = new Set(tags);
  let quant = null;
  for (const t of tags) if (QUANT_TAGS.has(t.toLowerCase())) { quant = t.toLowerCase(); break; }
  const nm = QUANT_NAME_RE.exec(m.id.split('/').pop() || '');
  if (nm) { const q = nm[1].toLowerCase(); quant = quant ? (quant === q ? quant : quant + '/' + q) : q; }
  if (!quant && gg) quant = 'gguf';
  if (!quant && rel.quantized) quant = 'quantized';
  // GGUF metadata on the Hub comes from one file in the repo; multi-file repos (draft heads, mmproj) mislead badly.
  if (!st && fromName && (!params || params < fromName / 2 || params > fromName * 2)) { params = fromName; paramsSrc = 'name'; }
  // Safetensors counts of packed quantized weights (AWQ/GPTQ int32 packing, MLX 4-bit) undercount by up to 8x.
  if (st && fromName && quant && st < fromName / 2) { params = fromName; paramsSrc = 'name'; }
  const files = (Array.isArray(m.siblings) ? m.siblings : []).map((x) => x.rfilename).filter(Boolean);
  const ggufAll = files.filter((f) => /\.gguf$/i.test(f));
  const ggufMain = ggufAll.filter((f) => !isAuxGguf(f));
  const quants = [...new Set(ggufMain.map(quantLabel).filter(Boolean))].sort((a, b) => (bpwFor(b) || 0) - (bpwFor(a) || 0));
  const ipm = Array.isArray(m.inferenceProviderMapping) ? m.inferenceProviderMapping : [];
  const providersLive = ipm.filter((p) => p.status === 'live').map((p) => p.provider);
  const ageDays = daysAgo(created, now);
  const dlAll = m.downloadsAllTime ?? null;
  const dl30 = m.downloads ?? null;
  return {
    id: m.id, author: m.author || m.id.split('/')[0], name: m.id.split('/').pop(),
    gated: m.gated && m.gated !== false ? String(m.gated) : null,
    likes: m.likes ?? 0, trending: m.trendingScore ?? 0, downloads: dl30, downloadsAllTime: dlAll,
    created, modified, ageDays,
    params, paramsSrc, paramsMeta: st || (gg && gg.total) || null,
    dtypes: m.safetensors ? Object.keys(m.safetensors.parameters || {}) : [],
    license, langs, datasets, arxiv, arch, archs,
    library: m.library_name || null, pipeline: m.pipeline_tag || null,
    tags, tagSet, hasSafetensors: !!m.safetensors || files.some((f) => /\.safetensors$/i.test(f)), hasGguf: !!gg || ggufAll.length > 0,
    ggufSize: gg ? gg.totalFileSize ?? null : null, ggufSizeMax: null, ctx: gg ? gg.context_length ?? null : null,
    files, ggufMain, ggufAux: ggufAll.filter(isAuxGguf), quants, hasMmproj: ggufAll.some((f) => /mmproj|projector/i.test(f)),
    fileCheck: null, repoBytes: null,
    quant, rel, relation, baseIds, providersLive: providersLive.length, providers: providersLive,
    customCode: tagSet.has('custom_code'),
    momentum: dlAll ? Math.min(1, (dl30 || 0) / dlAll) : null,
    dlPerDay: dlAll != null && ageDays > 0 ? dlAll / Math.max(ageDays, 1) : null,
    likesPerK: dlAll ? (m.likes || 0) / (dlAll / 10000) : null,
  };
}

// ---------- IndexedDB cache ----------
// Raw API results are cached per fetch signature so repeated queries load instantly and offline.
// Unavailable storage (Safari on file://, private windows, blocked site data) silently degrades to network-only.
const CACHE_DB = 'hf-explorer', CACHE_STORE = 'fetches', FILES_STORE = 'files', CACHE_MAX = 25;
let dbPromise = null;
function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((res) => {
    try {
      if (typeof indexedDB === 'undefined') return res(null);
      const r = indexedDB.open(CACHE_DB, 2);
      r.onupgradeneeded = () => {
        const db = r.result;
        if (!db.objectStoreNames.contains(CACHE_STORE)) db.createObjectStore(CACHE_STORE, { keyPath: 'key' }).createIndex('fetchedAt', 'fetchedAt');
        if (!db.objectStoreNames.contains(FILES_STORE)) db.createObjectStore(FILES_STORE, { keyPath: 'key' });
      };
      r.onsuccess = () => res(r.result);
      r.onerror = r.onblocked = () => res(null);
    } catch { res(null); }
  });
  return dbPromise;
}
function idb(mode, fn, store = CACHE_STORE) {
  return openDb().then((db) => db ? new Promise((res, rej) => {
    try {
      const tx = db.transaction(store, mode);
      const req = fn(tx.objectStore(store));
      tx.oncomplete = () => res(req ? req.result : undefined);
      tx.onerror = tx.onabort = () => rej(tx.error);
    } catch (e) { rej(e); }
  }) : undefined);
}
const cacheGet = (key) => idb('readonly', (s) => s.get(key)).catch(() => undefined);
async function cachePut(entry) {
  try {
    await idb('readwrite', (s) => s.put(entry));
    const keys = await idb('readonly', (s) => s.index('fetchedAt').getAllKeys());   // oldest first
    if (keys && keys.length > CACHE_MAX) await idb('readwrite', (s) => { for (const k of keys.slice(0, keys.length - CACHE_MAX)) s.delete(k); });
  } catch (e) { console.warn('cache write failed', e); }
  updateCacheInfo();
}
const cacheClear = () => Promise.all([idb('readwrite', (s) => s.clear()), idb('readwrite', (s) => s.clear(), FILES_STORE)]).catch(() => undefined).then(updateCacheInfo);
const fcKey = (r) => r.id + '@' + (r.modified ? r.modified.toISOString() : '');
const fcGet = (r) => idb('readonly', (s) => s.get(fcKey(r)), FILES_STORE).catch(() => undefined);
const fcPut = (r, info) => idb('readwrite', (s) => s.put({ key: fcKey(r), ...info }), FILES_STORE).catch(() => undefined);
const fcAll = () => idb('readonly', (s) => s.getAll(), FILES_STORE).catch(() => undefined);
async function updateCacheInfo() {
  const el = $('cacheInfo');
  const n = await idb('readonly', (s) => s.count()).catch(() => undefined);
  if (n === undefined) { el.textContent = '(no local cache available in this browser)'; return; }
  const nf = await idb('readonly', (s) => s.count(), FILES_STORE).catch(() => 0);
  let size = '';
  try { const est = await navigator.storage.estimate(); if (est.usage) size = ', ' + fmtBytes(est.usage); } catch { /* ignore */ }
  el.innerHTML = n || nf ? `cache: ${n} quer${n === 1 ? 'y' : 'ies'}, ${nf} file check${nf === 1 ? '' : 's'}${size} · <button type="button" class="link" id="clearCache">clear</button>` : '';
  el.querySelector('#clearCache')?.addEventListener('click', cacheClear);
}
const fmtAgo = (ms) => { const m = ms / 60000; return m < 1 ? 'just now' : m < 60 ? `${Math.round(m)} min ago` : m < 48 * 60 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} days ago`; };

// ---------- Fetching ----------
function fetchParams() {
  return {
    search: $('f_search').value.trim(), author: $('f_author').value.trim(), pipeline: $('f_pipeline').value,
    library: $('f_library').value, filter: $('f_filter').value.trim(), sort: $('f_sort').value,
    limit: +$('f_limit').value, config: $('f_config').checked, gguf: $('f_gguf').checked, inference: $('f_inference').checked,
  };
}
function buildUrl(p, cursor) {
  const u = new URL(API);
  const q = u.searchParams;
  if (p.search) q.set('search', p.search);
  if (p.author) q.set('author', p.author);
  if (p.pipeline) q.set('pipeline_tag', p.pipeline);
  if (p.library) q.set('library', p.library);
  for (const t of p.filter.split(',').map((s) => s.trim()).filter(Boolean)) q.append('filter', t);
  if (p.gguf) q.set('gguf', 'true');
  if (p.inference) q.set('inference_provider', 'all');
  q.set('sort', p.sort);
  q.set('direction', '-1');
  q.set('limit', String(Math.min(PAGE, p.limit)));
  for (const e of EXPAND) q.append('expand[]', e);
  if (p.config) q.append('expand[]', 'config');
  if (cursor) q.set('cursor', cursor);
  return u.toString();
}
function nextCursor(res) {
  const link = res.headers.get('link');
  if (!link) return null;
  const m = /<([^>]+)>;\s*rel="next"/.exec(link);
  if (!m) return null;
  try { return new URL(m[1]).searchParams.get('cursor'); } catch { return null; }
}
async function loadRows(raw, key, fetchedAt, note) {
  const now = Date.now();
  state.rows = raw.map((m) => normalize(m, now));
  const cached = await fcAll();
  if (cached && cached.length) {
    const byKey = new Map(cached.map((c) => [c.key, c]));
    for (const r of state.rows) { const c = byKey.get(fcKey(r)); if (c) applyFileCheck(r, c); }
  }
  state.fetchedWith = key;
  state.open.clear();
  state.page = 0;
  const age = now - fetchedAt;
  const ageNote = age > 60000 ? ` (fetched ${fmtAgo(age)}${age > 86400000 ? ', press refresh for current numbers' : ''})` : '';
  setStatus(`${state.rows.length.toLocaleString()} models ${note}${ageNote}.`);
  buildFacets();
  apply();
}
async function doFetch(force) {
  if (state.abort) state.abort.abort();
  const p = fetchParams();
  const key = JSON.stringify(p);
  if (!force) {
    const hit = await cacheGet(key);
    if (hit && Array.isArray(hit.raw)) { await loadRows(hit.raw, key, hit.fetchedAt, 'loaded from local cache'); return; }
  }
  const ctrl = new AbortController();
  state.abort = ctrl;
  $('fetchBtn').disabled = true;
  $('cancelBtn').hidden = false;
  setStatus('Fetching…');
  const all = [];
  const seen = new Set();
  let cursor = null;
  const t0 = performance.now();
  try {
    while (all.length < p.limit) {
      const res = await fetch(buildUrl(p, cursor), { signal: ctrl.signal, headers: { Accept: 'application/json' } });
      if (!res.ok) {
        let msg = res.status + ' ' + res.statusText;
        try { msg += ': ' + (await res.json()).error; } catch { /* ignore */ }
        throw new Error(msg);
      }
      const batch = await res.json();
      for (const m of batch) if (!seen.has(m.id)) { seen.add(m.id); all.push(m); }
      setStatus(`Fetched ${all.length.toLocaleString()} models… (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
      cursor = nextCursor(res);
      if (!cursor || batch.length === 0) break;
    }
    const raw = all.slice(0, p.limit);
    const fetchedAt = Date.now();
    await loadRows(raw, key, fetchedAt, `fetched in ${((performance.now() - t0) / 1000).toFixed(1)}s, ordered by ${p.sort} server-side${cursor ? ', more available' : ', exhausted'}`);
    cachePut({ key, params: p, fetchedAt, count: raw.length, raw });
  } catch (e) {
    if (e.name === 'AbortError') setStatus(`Cancelled. Kept ${state.rows.length.toLocaleString()} previously fetched models.`);
    else setStatus('Error: ' + e.message, true);
  } finally {
    $('fetchBtn').disabled = false;
    $('cancelBtn').hidden = true;
    state.abort = null;
  }
}
function setStatus(msg, err) { const s = $('status'); s.textContent = msg; s.classList.toggle('err', !!err); }

// ---------- File checks (tree API) and exact GGUF header reads ----------
async function fetchTree(id, signal) {
  const out = [];
  let url = `https://huggingface.co/api/models/${id}/tree/main?recursive=true&limit=1000`;
  for (let page = 0; url && page < 5; page++) {
    const res = await fetch(url, { signal, headers: { Accept: 'application/json' } });
    if (res.status === 429) throw new Error('rate limited by the Hub (429); wait a few minutes');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    out.push(...(await res.json()));
    url = nextCursor(res) ? url.replace(/&cursor=[^&]*/, '') + '&cursor=' + encodeURIComponent(nextCursor(res)) : null;
  }
  return out;
}
function summarizeTree(tree) {
  const files = tree.filter((f) => f.type === 'file');
  const info = { checkedAt: Date.now(), totalBytes: 0, stBytes: 0, quants: {}, auxBytes: 0, nFiles: files.length };
  for (const f of files) {
    const size = f.size || 0;
    info.totalBytes += size;
    if (/\.gguf$/i.test(f.path)) {
      if (isAuxGguf(f.path)) { info.auxBytes += size; continue; }
      const q = quantLabel(f.path) || 'other';
      info.quants[q] = (info.quants[q] || 0) + size;
    } else if (/\.safetensors$/i.test(f.path) && !/mmproj|projector/i.test(f.path)) info.stBytes += size;
  }
  let best = null;
  for (const [q, bytes] of Object.entries(info.quants)) { const b = bpwFor(q); if (b && (!best || b > best.bpw)) best = { q, bytes, bpw: b }; }
  if (best) { info.estParams = Math.round(best.bytes * 8 / best.bpw); info.estFrom = best.q; }
  const sizes = Object.values(info.quants).filter((b) => b > 0);
  if (sizes.length) { info.ggufMin = Math.min(...sizes); info.ggufMax = Math.max(...sizes); }
  return info;
}
function applyFileCheck(r, info) {
  r.fileCheck = info;
  r.repoBytes = info.totalBytes;
  if (info.ggufMin) { r.ggufSize = info.ggufMin; r.ggufSizeMax = info.ggufMax; }
  const q = Object.keys(info.quants || {}).filter((k) => k !== 'other');
  if (q.length) r.quants = q.sort((a, b) => (bpwFor(b) || 0) - (bpwFor(a) || 0));
  if (info.header && info.header.params) {
    r.params = info.header.params; r.paramsSrc = 'header';
    if (info.header.contextLength) r.ctx = info.header.contextLength;
  } else if (r.paramsSrc !== 'safetensors' && info.estParams) {
    r.params = info.estParams; r.paramsSrc = 'files';
  }
}
async function checkFiles(r, signal) {
  const cached = await fcGet(r);
  if (cached && cached.checkedAt) { applyFileCheck(r, cached); return cached; }
  const info = summarizeTree(await fetchTree(r.id, signal));
  await fcPut(r, info);
  applyFileCheck(r, info);
  return info;
}
function pickHeaderFile(r) {
  // Smallest main GGUF (cheapest to range-read; the header is the same for every quant). For shards, the first one.
  const cands = r.ggufMain.filter((f) => !/-\d{5}-of-\d{5}\.gguf$/i.test(f) || /-00001-of-/i.test(f));
  if (!cands.length) return null;
  const sizes = r.fileCheck ? r.fileCheck.quants : {};
  return cands.map((f) => ({ f, s: sizes[quantLabel(f) || 'other'] || (bpwFor(quantLabel(f) || '') || 99) * 1e9 }))
    .sort((a, b) => a.s - b.s)[0].f;
}
async function readHeader(r, signal) {
  const file = pickHeaderFile(r);
  if (!file) throw new Error('no main GGUF file in this repo');
  const h = await GGUF.readHeader(`https://huggingface.co/${r.id}/resolve/main/${file.split('/').map(encodeURIComponent).join('/')}`, { signal });
  const header = { file, params: h.params, nTensors: h.nTensors, arch: h.arch, name: h.name, sizeLabel: h.sizeLabel, fileType: h.fileType,
    contextLength: h.contextLength, blockCount: h.blockCount, expertCount: h.expertCount, headerBytes: h.headerBytes, readAt: Date.now() };
  const info = { ...(r.fileCheck || (await fcGet(r)) || { quants: {}, totalBytes: null }), header };
  await fcPut(r, info);
  applyFileCheck(r, info);
  return header;
}
async function checkFilesBatch() {
  const btn = $('checkFiles');
  if (state.fcAbort) { state.fcAbort.abort(); return; }
  const todo = state.view.filter((r) => !r.fileCheck || !r.fileCheck.checkedAt).slice(0, 300);
  if (!todo.length) { setStatus('All filtered models already have file checks.'); return; }
  const ctrl = new AbortController();
  state.fcAbort = ctrl;
  btn.textContent = 'stop';
  let done = 0, failed = 0;
  const t0 = performance.now();
  const worker = async () => {
    while (todo.length && !ctrl.signal.aborted) {
      const r = todo.shift();
      try { await checkFiles(r, ctrl.signal); done++; }
      catch (e) { if (e.name === 'AbortError') return; failed++; if (/429/.test(e.message)) { ctrl.abort(); setStatus('Stopped: ' + e.message, true); return; } }
      setStatus(`Checking files… ${done} done${failed ? `, ${failed} failed` : ''}, ${todo.length} left`);
    }
  };
  await Promise.all([1, 2, 3, 4].map(worker));
  if (!ctrl.signal.aborted || !/Stopped/.test($('status').textContent)) setStatus(`File check: ${done} models in ${((performance.now() - t0) / 1000).toFixed(1)}s${failed ? `, ${failed} failed` : ''}${ctrl.signal.aborted ? ' (stopped)' : ''}.`);
  state.fcAbort = null;
  btn.textContent = 'check files';
  apply();
  updateCacheInfo();
}

// ---------- Facets ----------
const FACETS = [
  { key: 'license', label: 'license', get: (r) => r.license ? [r.license] : ['(none)'] },
  { key: 'arch', label: 'architecture', get: (r) => [r.arch || '(unknown)'] },
  { key: 'pipeline', label: 'task', get: (r) => [r.pipeline || '(none)'] },
  { key: 'library', label: 'library', get: (r) => [r.library || '(none)'] },
  { key: 'quant', label: 'quantization', get: (r) => [r.quant || '(none)'] },
  { key: 'quants', label: 'gguf quant available', get: (r) => r.quants.length ? r.quants : ['(none)'] },
  { key: 'dtypes', label: 'dtype (safetensors)', get: (r) => r.dtypes.length ? r.dtypes : ['(unknown)'] },
  { key: 'langs', label: 'language', get: (r) => r.langs.length ? r.langs : ['(none)'] },
  { key: 'author', label: 'author', get: (r) => [r.author] },
  { key: 'providers', label: 'inference provider', get: (r) => r.providers.length ? r.providers : ['(none)'] },
  { key: 'base', label: 'base model', get: (r) => r.baseIds.length ? r.baseIds : ['(none)'] },
];
function buildFacets() {
  const host = $('facets');
  host.innerHTML = '';
  for (const f of FACETS) {
    const counts = new Map();
    for (const r of state.rows) for (const v of f.get(r)) counts.set(v, (counts.get(v) || 0) + 1);
    const sel = state.facetSel[f.key] || (state.facetSel[f.key] = new Set());
    for (const v of [...sel]) if (!counts.has(v)) sel.delete(v);
    const det = document.createElement('details');
    det.className = 'facet';
    det.open = sel.size > 0 || ['license', 'arch'].includes(f.key);
    det.innerHTML = `<summary>${f.label} <span class="muted">(${counts.size})</span></summary>` +
      (counts.size > 8 ? `<input class="facet-search" type="search" placeholder="filter ${f.label}…">` : '') + `<div class="facet-list"></div>`;
    host.appendChild(det);
    const list = det.querySelector('.facet-list');
    const render = (needle) => {
      const entries = [...counts.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
      list.innerHTML = entries.filter(([v]) => !needle || String(v).toLowerCase().includes(needle))
        .slice(0, 300)
        .map(([v, n]) => `<label><input type="checkbox" data-f="${f.key}" data-v="${esc(v)}" ${sel.has(v) ? 'checked' : ''}><span class="v" title="${esc(v)}">${esc(v)}</span><span class="n">${fmtInt(n)}</span></label>`).join('');
    };
    render('');
    det.querySelector('.facet-search')?.addEventListener('input', (e) => render(e.target.value.toLowerCase()));
    list.addEventListener('change', (e) => {
      const cb = e.target;
      if (cb.checked) sel.add(cb.dataset.v); else sel.delete(cb.dataset.v);
      state.page = 0; apply();
    });
  }
}

// ---------- Filtering ----------
function readFilters() {
  const v = (id) => $(id).value.trim();
  const num = (id) => { const x = v(id); return x === '' ? null : +x; };
  const date = (id) => { const x = v(id); return x ? new Date(x + 'T00:00:00Z').getTime() : null; };
  const re = (id) => { const x = v(id); if (!x) return null; try { return new RegExp(x, 'i'); } catch { return null; } };
  const list = (id) => v(id).split(',').map((s) => s.trim()).filter(Boolean);
  return {
    nameInc: re('c_nameInc'), nameExc: re('c_nameExc'),
    pMin: num('c_pMin'), pMax: num('c_pMax'), pUnknown: $('c_pUnknown').checked,
    dlMin: num('c_dlMin'), dlAllMin: num('c_dlAllMin'), likesMin: num('c_likesMin'), trendMin: num('c_trendMin'),
    ctxMin: num('c_ctxMin'), provMin: num('c_provMin'),
    createdAfter: date('c_createdAfter'), createdBefore: date('c_createdBefore'), modAfter: date('c_modAfter'), modBefore: date('c_modBefore'),
    gated: v('c_gated'), quant: v('c_quant'), st: v('c_st'), gg: v('c_gg'), relation: v('c_relation'), customCode: v('c_customCode'),
    live: $('c_live').checked, tagsInc: list('c_tagsInc'), tagsExc: list('c_tagsExc'),
  };
}
function passes(r, f) {
  if (f.nameInc && !f.nameInc.test(r.id)) return false;
  if (f.nameExc && f.nameExc.test(r.id)) return false;
  if (f.pMin != null || f.pMax != null) {
    if (r.params == null) { if (!f.pUnknown) return false; }
    else {
      if (f.pMin != null && r.params < f.pMin * 1e9) return false;
      if (f.pMax != null && r.params > f.pMax * 1e9) return false;
    }
  }
  if (f.dlMin != null && (r.downloads ?? 0) < f.dlMin) return false;
  if (f.dlAllMin != null && (r.downloadsAllTime ?? 0) < f.dlAllMin) return false;
  if (f.likesMin != null && r.likes < f.likesMin) return false;
  if (f.trendMin != null && r.trending < f.trendMin) return false;
  if (f.ctxMin != null && (r.ctx ?? 0) < f.ctxMin) return false;
  if (f.provMin != null && r.providersLive < f.provMin) return false;
  if (f.createdAfter != null && (!r.created || r.created.getTime() < f.createdAfter)) return false;
  if (f.createdBefore != null && (!r.created || r.created.getTime() > f.createdBefore)) return false;
  if (f.modAfter != null && (!r.modified || r.modified.getTime() < f.modAfter)) return false;
  if (f.modBefore != null && (!r.modified || r.modified.getTime() > f.modBefore)) return false;
  if (f.gated === 'yes' && !r.gated) return false;
  if (f.gated === 'no' && r.gated) return false;
  if (f.quant === 'yes' && !r.quant) return false;
  if (f.quant === 'no' && r.quant) return false;
  if (f.st === 'yes' && !r.hasSafetensors) return false;
  if (f.st === 'no' && r.hasSafetensors) return false;
  if (f.gg === 'yes' && !r.hasGguf) return false;
  if (f.gg === 'no' && r.hasGguf) return false;
  if (f.relation === 'none' && r.baseIds.length) return false;
  if (f.relation === 'any' && !r.baseIds.length) return false;
  if (['finetune', 'quantized', 'adapter', 'merge'].includes(f.relation) && !r.rel[f.relation]) return false;
  if (f.customCode === 'yes' && !r.customCode) return false;
  if (f.customCode === 'no' && r.customCode) return false;
  if (f.live && r.providersLive === 0) return false;
  for (const t of f.tagsInc) if (!r.tagSet.has(t)) return false;
  for (const t of f.tagsExc) if (r.tagSet.has(t)) return false;
  for (const fc of FACETS) {
    const sel = state.facetSel[fc.key];
    if (!sel || sel.size === 0) continue;
    if (!fc.get(r).some((v) => sel.has(v))) return false;
  }
  return true;
}
function countActiveFilters(f) {
  let n = 0;
  for (const [k, v] of Object.entries(f)) {
    if (k === 'pUnknown') continue;
    if (Array.isArray(v) ? v.length : v !== null && v !== '' && v !== false) n++;
  }
  for (const s of Object.values(state.facetSel)) if (s.size) n++;
  return n;
}

// ---------- Sorting ----------
function cmpVal(a, b, type) {
  if (type === 'str') return String(a).localeCompare(String(b), undefined, { sensitivity: 'base' });
  if (type === 'date') return a.getTime() - b.getTime();
  return a - b;
}
function sortView() {
  const keys = state.sort.map((s) => ({ ...s, col: COL[s.key] })).filter((s) => s.col);
  state.view.sort((x, y) => {
    for (const s of keys) {
      const a = s.col.get(x), b = s.col.get(y);
      const an = a == null || a === '', bn = b == null || b === '';
      if (an || bn) { if (an && bn) continue; return an ? 1 : -1; }   // nulls last regardless of direction
      const c = cmpVal(a, b, s.col.type);
      if (c) return c * s.dir;
    }
    return 0;
  });
}

// ---------- Render ----------
function apply() {
  const f = readFilters();
  state.view = state.rows.filter((r) => passes(r, f));
  sortView();
  const n = countActiveFilters(f);
  $('filterCount').textContent = n ? `${n} active` : '';
  renderTable();
  syncUrl();
}
function renderHead() {
  const cols = COLUMNS.filter((c) => state.cols.has(c.key));
  $('headRow').innerHTML = cols.map((c) => {
    const i = state.sort.findIndex((s) => s.key === c.key);
    const dir = i >= 0 ? `<span class="dir">${state.sort[i].dir < 0 ? '▼' : '▲'}${state.sort.length > 1 ? i + 1 : ''}</span>` : '';
    return `<th data-k="${c.key}" class="${c.type === 'num' ? 'num' : ''}" title="${esc(c.title || '')}">${esc(c.label)}${dir}</th>`;
  }).join('');
}
function cell(c, r) {
  const v = c.get(r);
  if (c.key === 'id') {
    const pills = (r.gated ? `<span class="pill gated">gated</span>` : '') + (r.providersLive ? `<span class="pill live">${r.providersLive} live</span>` : '');
    return `<td class="id"><a href="https://huggingface.co/${esc(r.id)}" target="_blank" rel="noopener">${esc(r.id)}</a>${pills}</td>`;
  }
  let txt = c.fmt ? c.fmt(v) : (v ?? '');
  if (c.key === 'params' && txt && (r.paramsSrc === 'name' || r.paramsSrc === 'files')) txt = '~' + txt;
  if (c.key === 'ggufSize' && r.ggufSizeMax && r.ggufSizeMax !== v) txt = `${fmtBytes(v)} – ${fmtBytes(r.ggufSizeMax)}`;
  const title = c.type === 'num' && v != null ? (Number.isInteger(v) ? fmtInt(v) : v.toFixed(3)) : txt;
  return `<td class="${c.type === 'num' ? 'num' : ''}" title="${esc(title)}">${esc(txt)}</td>`;
}
function detail(r) {
  const row = (k, v) => v ? `<dt>${k}</dt><dd>${v}</dd>` : '';
  const link = (id) => `<a href="https://huggingface.co/${esc(id)}" target="_blank" rel="noopener">${esc(id)}</a>`;
  const tags = r.tags.map((t) => `<span class="tag">${esc(t)}</span>`).join('');
  const rels = Object.entries(r.rel).map(([k, ids]) => `${k}: ${ids.map(link).join(', ')}`).join('<br>');
  const dt = r.dtypes.length ? r.dtypes.join(', ') : '';
  return `<div class="links"><a href="https://huggingface.co/${esc(r.id)}" target="_blank" rel="noopener">model page</a>` +
    `<a href="https://huggingface.co/${esc(r.id)}/tree/main" target="_blank" rel="noopener">files</a>` +
    `<a href="https://huggingface.co/api/models/${esc(r.id)}" target="_blank" rel="noopener">api json</a>` +
    `<a href="https://huggingface.co/models?other=base_model:finetune:${encodeURIComponent(r.id)}" target="_blank" rel="noopener">finetunes</a>` +
    `<a href="https://huggingface.co/models?other=base_model:quantized:${encodeURIComponent(r.id)}" target="_blank" rel="noopener">quantizations</a>` +
    `<button type="button" class="copy" data-id="${esc(r.id)}">copy id</button></div>` +
    `<dl class="detail-grid">` +
    row('params', paramsExplain(r)) +
    row('files', filesExplain(r)) +
    row('architectures', r.archs.length ? esc(r.archs.join(', ')) : '') +
    row('base models', rels) +
    row('gguf', r.hasGguf ? `${fmtBytes(r.ggufSize)}${r.ctx ? ', ctx ' + fmtInt(r.ctx) : ''}` : '') +
    row('providers', r.providers.length ? esc(r.providers.join(', ')) : '') +
    row('gated', r.gated ? esc(r.gated) : '') +
    row('datasets', r.datasets.map((d) => `<a href="https://huggingface.co/datasets/${esc(d)}" target="_blank" rel="noopener">${esc(d)}</a>`).join(', ')) +
    row('arxiv', r.arxiv.map((a) => `<a href="https://arxiv.org/abs/${esc(a)}" target="_blank" rel="noopener">${esc(a)}</a>`).join(', ')) +
    row('stats', `${fmtInt(r.downloads)} dl/30d · ${fmtInt(r.downloadsAllTime)} total · ${fmtInt(r.likes)} likes · trending ${fmtInt(r.trending)} · ${fmtCompact(r.dlPerDay)} dl/day · momentum ${fmtPct(r.momentum)}`) +
    row('dates', `created ${fmtDate(r.created)} · modified ${fmtDate(r.modified)} · ${Math.floor(r.ageDays ?? 0)} days old`) +
    row('tags', tags) + `</dl>`;
}
function paramsExplain(r) {
  const src = r.paramsSrc, fc = r.fileCheck;
  const metaNote = r.paramsMeta && r.paramsMeta !== r.params ? (r.hasSafetensors && !r.ggufMain.length
    ? ` Hub safetensors metadata says ${fmtInt(r.paramsMeta)}; packed quantized weights (AWQ/GPTQ/MLX) are undercounted there.`
    : ` Hub GGUF metadata says ${fmtInt(r.paramsMeta)}, which describes a single file in the repo (e.g. a draft/MTP or mmproj GGUF).`) : '';
  if (!r.params) return src ? '' : 'unknown; no size metadata, no size in the name' + (r.ggufMain.length ? ', read the GGUF header below' : '');
  if (src === 'header') return `<b>${fmtInt(r.params)}</b> exact, from the GGUF header of <code>${esc(basename(fc.header.file))}</code> (${fmtInt(fc.header.nTensors)} tensors${fc.header.sizeLabel ? ', size label ' + esc(fc.header.sizeLabel) : ''}${fc.header.fileType ? ', ' + esc(fc.header.fileType) : ''}${fc.header.expertCount ? ', ' + fc.header.expertCount + ' experts' : ''}).${metaNote}`;
  if (src === 'files') return `~${fmtInt(r.params)} estimated from the ${esc(fc.estFrom)} file size at ${bpwFor(fc.estFrom)} bits/weight.${metaNote}`;
  if (src === 'name') return `~${fmtInt(r.params)} parsed from the name.${metaNote || ' No size metadata on the Hub.'}`;
  const dt = r.dtypes.length ? ', ' + r.dtypes.join(', ') : '';
  return `${fmtInt(r.params)} (${src}${dt})`;
}
function filesExplain(r) {
  const fc = r.fileCheck;
  let html = '';
  if (fc && fc.checkedAt) {
    const rows = Object.entries(fc.quants).sort((a, b) => b[1] - a[1]).map(([q, b]) => `<tr><td>${esc(q)}</td><td class="num">${fmtBytes(b)}</td></tr>`).join('');
    html += `repo ${fmtBytes(fc.totalBytes)} in ${fc.nFiles} files` + (fc.stBytes ? ` · safetensors weights ${fmtBytes(fc.stBytes)}` : '') + (fc.auxBytes ? ` · auxiliary GGUF (mmproj/draft) ${fmtBytes(fc.auxBytes)}` : '') +
      (rows ? `<table class="files">${rows}</table>` : '');
  } else {
    if (r.quants.length) html += `GGUF quants: ${r.quants.map((q) => `<span class="tag">${esc(q)}</span>`).join('')}`;
    if (r.ggufAux.length) html += `<div class="muted">auxiliary GGUF: ${r.ggufAux.map((f) => esc(basename(f))).join(', ')}</div>`;
    html += `<div class="muted">${r.files.length} files listed</div>`;
  }
  const btns = (fc && fc.checkedAt ? '' : `<button type="button" class="act" data-act="check" data-id="${esc(r.id)}">check files (sizes)</button>`) +
    (r.ggufMain.length && !(fc && fc.header) ? `<button type="button" class="act" data-act="header" data-id="${esc(r.id)}">read GGUF header (exact params, ~4–12 MB)</button>` : '');
  return html + (btns ? `<div>${btns}<span class="act-msg"></span></div>` : '');
}
function renderTable() {
  renderHead();
  const cols = COLUMNS.filter((c) => state.cols.has(c.key));
  const total = state.view.length;
  const pages = Math.max(1, Math.ceil(total / state.pageSize));
  state.page = Math.min(state.page, pages - 1);
  const start = state.page * state.pageSize;
  const slice = state.view.slice(start, start + state.pageSize);
  const body = $('body');
  if (!state.rows.length) {
    body.innerHTML = `<tr><td colspan="${cols.length}" class="empty">Nothing fetched yet.</td></tr>`;
  } else if (!total) {
    body.innerHTML = `<tr><td colspan="${cols.length}" class="empty">No models match the current filters.</td></tr>`;
  } else {
    let html = '';
    for (const r of slice) {
      const open = state.open.has(r.id);
      html += `<tr class="row${open ? ' open' : ''}" data-id="${esc(r.id)}">${cols.map((c) => cell(c, r)).join('')}</tr>`;
      if (open) html += `<tr class="detail"><td colspan="${cols.length}">${detail(r)}</td></tr>`;
    }
    body.innerHTML = html;
  }
  $('summary').innerHTML = state.rows.length
    ? `<b>${fmtInt(total)}</b> of ${fmtInt(state.rows.length)} models match · ${aggregate(state.view)}`
    : 'No data yet. Set the fetch options above and press Fetch.';
  const pg = $('pager');
  pg.innerHTML = total > state.pageSize
    ? `<button type="button" data-p="first" ${state.page === 0 ? 'disabled' : ''}>«</button><button type="button" data-p="prev" ${state.page === 0 ? 'disabled' : ''}>‹</button>` +
      `<span>page ${state.page + 1} / ${pages}</span>` +
      `<button type="button" data-p="next" ${state.page >= pages - 1 ? 'disabled' : ''}>›</button><button type="button" data-p="last" ${state.page >= pages - 1 ? 'disabled' : ''}>»</button>` +
      `<span class="muted">rows ${fmtInt(start + 1)}–${fmtInt(Math.min(total, start + state.pageSize))}</span>`
    : `<span class="muted">${fmtInt(total)} rows</span>`;
}
function aggregate(rows) {
  if (!rows.length) return '';
  let dl = 0, likes = 0;
  const ps = [];
  for (const r of rows) { dl += r.downloads || 0; likes += r.likes; if (r.params) ps.push(r.params); }
  ps.sort((a, b) => a - b);
  const med = ps.length ? fmtParams(ps[Math.floor(ps.length / 2)]) : '–';
  return `${fmtCompact(dl)} dl/30d · ${fmtCompact(likes)} likes · median size ${med} (${ps.length} with known size)`;
}
function renderColPicker() {
  $('colPicker').innerHTML = COLUMNS.map((c) => `<label><input type="checkbox" data-c="${c.key}" ${state.cols.has(c.key) ? 'checked' : ''}> ${esc(c.label)}</label>`).join('');
}

// ---------- Export ----------
function exportRows(kind) {
  const cols = COLUMNS.filter((c) => state.cols.has(c.key));
  let blob, name;
  if (kind === 'json') {
    const data = state.view.map((r) => { const { tagSet, ...rest } = r; return rest; });
    blob = new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' });
    name = 'hf-models.json';
  } else {
    const q = (s) => { s = s == null ? '' : String(s); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
    const raw = (c, r) => { const v = c.get(r); return v instanceof Date ? v.toISOString() : v; };
    const lines = [cols.map((c) => q(c.label)).join(',')];
    for (const r of state.view) lines.push(cols.map((c) => q(raw(c, r))).join(','));
    blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    name = 'hf-models.csv';
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// ---------- URL state ----------
const FETCH_IDS = ['f_search', 'f_author', 'f_pipeline', 'f_library', 'f_filter', 'f_sort', 'f_limit', 'f_config', 'f_gguf', 'f_inference'];
const FILTER_IDS = ['c_nameInc', 'c_nameExc', 'c_pMin', 'c_pMax', 'c_pUnknown', 'c_dlMin', 'c_dlAllMin', 'c_likesMin', 'c_trendMin', 'c_ctxMin', 'c_provMin',
  'c_createdAfter', 'c_createdBefore', 'c_modAfter', 'c_modBefore', 'c_gated', 'c_quant', 'c_st', 'c_gg', 'c_relation', 'c_customCode', 'c_live', 'c_tagsInc', 'c_tagsExc'];
const DEFAULTS = {};
function snapshotDefaults() { for (const id of [...FETCH_IDS, ...FILTER_IDS]) { const el = $(id); DEFAULTS[id] = el.type === 'checkbox' ? el.checked : el.value; } }
function syncUrl() {
  const q = new URLSearchParams();
  for (const id of [...FETCH_IDS, ...FILTER_IDS]) {
    const el = $(id);
    const v = el.type === 'checkbox' ? el.checked : el.value;
    if (v !== DEFAULTS[id]) q.set(id, el.type === 'checkbox' ? (v ? '1' : '0') : v);
  }
  for (const [k, s] of Object.entries(state.facetSel)) if (s.size) q.set('fx_' + k, [...s].join(SEP));
  q.set('sort', state.sort.map((s) => (s.dir < 0 ? '-' : '') + s.key).join(','));
  const cols = [...state.cols].join(',');
  if (cols !== COLUMNS.filter((c) => c.on).map((c) => c.key).join(',')) q.set('cols', cols);
  history.replaceState(null, '', '?' + q.toString());
}
function loadUrl() {
  const q = new URLSearchParams(location.search);
  let any = false;
  for (const id of [...FETCH_IDS, ...FILTER_IDS]) {
    if (!q.has(id)) continue;
    const el = $(id);
    if (el.type === 'checkbox') el.checked = q.get(id) === '1'; else el.value = q.get(id);
    if (FETCH_IDS.includes(id)) any = true;
  }
  for (const [k, v] of q.entries()) if (k.startsWith('fx_')) state.facetSel[k.slice(3)] = new Set(v.split(SEP));
  if (q.get('sort')) state.sort = q.get('sort').split(',').filter(Boolean).map((s) => s.startsWith('-') ? { key: s.slice(1), dir: -1 } : { key: s, dir: 1 }).filter((s) => COL[s.key]);
  if (q.get('cols')) state.cols = new Set(q.get('cols').split(',').filter((k) => COL[k]));
  return any || q.has('sort');
}

// ---------- Wiring ----------
function init() {
  for (const p of PIPELINES) $('f_pipeline').insertAdjacentHTML('beforeend', `<option value="${p}">${p}</option>`);
  for (const l of LIBRARIES) $('f_library').insertAdjacentHTML('beforeend', `<option value="${l}">${l}</option>`);
  snapshotDefaults();
  const auto = loadUrl();
  renderColPicker();
  renderTable();

  $('fetchForm').addEventListener('submit', (e) => { e.preventDefault(); doFetch(false); });
  $('refreshBtn').addEventListener('click', () => doFetch(true));
  updateCacheInfo();
  $('cancelBtn').addEventListener('click', () => state.abort?.abort());
  for (const id of FETCH_IDS) $(id).addEventListener('change', () => {
    if (state.rows.length && JSON.stringify(fetchParams()) !== state.fetchedWith) setStatus('Fetch options changed — press Fetch to reload the result set.');
  });

  let t;
  for (const id of FILTER_IDS) $(id).addEventListener('input', () => { clearTimeout(t); t = setTimeout(() => { state.page = 0; apply(); }, 150); });
  $('resetFilters').addEventListener('click', () => {
    for (const id of FILTER_IDS) { const el = $(id); if (el.type === 'checkbox') el.checked = DEFAULTS[id]; else el.value = DEFAULTS[id]; }
    for (const s of Object.values(state.facetSel)) s.clear();
    buildFacets(); state.page = 0; apply();
  });
  document.querySelector('.quick').addEventListener('click', (e) => {
    const d = e.target.dataset.days; if (!d) return;
    $('c_createdAfter').value = new Date(Date.now() - d * 86400000).toISOString().slice(0, 10);
    $('c_createdBefore').value = '';
    state.page = 0; apply();
  });

  $('headRow').addEventListener('click', (e) => {
    const th = e.target.closest('th'); if (!th) return;
    const key = th.dataset.k;
    const col = COL[key];
    const i = state.sort.findIndex((s) => s.key === key);
    if (e.shiftKey) {
      if (i >= 0) state.sort[i].dir *= -1; else state.sort.push({ key, dir: col.type === 'str' ? 1 : -1 });
    } else if (i === 0 && state.sort.length === 1) {
      state.sort[0].dir *= -1;
    } else {
      state.sort = [{ key, dir: col.type === 'str' ? 1 : -1 }];
    }
    sortView(); state.page = 0; renderTable(); syncUrl();
  });
  $('body').addEventListener('click', (e) => {
    if (e.target.closest('a')) return;
    const cp = e.target.closest('button.copy');
    if (cp) { navigator.clipboard?.writeText(cp.dataset.id); cp.textContent = 'copied'; return; }
    const act = e.target.closest('button.act');
    if (act) {
      const r = state.rows.find((x) => x.id === act.dataset.id); if (!r) return;
      act.disabled = true; act.textContent = act.dataset.act === 'check' ? 'checking…' : 'reading header…';
      (act.dataset.act === 'check' ? checkFiles(r) : readHeader(r))
        .then(() => { apply(); updateCacheInfo(); })
        .catch((err) => { act.disabled = false; act.textContent = 'retry'; const m = act.parentElement.querySelector('.act-msg'); if (m) { m.textContent = ' ' + err.message; m.className = 'act-msg err'; } });
      return;
    }
    const tr = e.target.closest('tr.row'); if (!tr) return;
    const id = tr.dataset.id;
    if (state.open.has(id)) state.open.delete(id); else state.open.add(id);
    renderTable();
  });
  $('pager').addEventListener('click', (e) => {
    const p = e.target.dataset.p; if (!p) return;
    const pages = Math.ceil(state.view.length / state.pageSize);
    state.page = p === 'first' ? 0 : p === 'prev' ? state.page - 1 : p === 'next' ? state.page + 1 : pages - 1;
    renderTable(); document.querySelector('.tablewrap').scrollTop = 0;
  });
  $('pageSize').addEventListener('change', (e) => { state.pageSize = +e.target.value; state.page = 0; renderTable(); });
  $('colPicker').addEventListener('change', (e) => {
    const k = e.target.dataset.c; if (!k) return;
    if (e.target.checked) state.cols.add(k); else state.cols.delete(k);
    renderTable(); syncUrl();
  });
  $('checkFiles').addEventListener('click', checkFilesBatch);
  $('exportCsv').addEventListener('click', () => exportRows('csv'));
  $('exportJson').addEventListener('click', () => exportRows('json'));
  $('copyLink').addEventListener('click', async (e) => {
    syncUrl();
    try { await navigator.clipboard.writeText(location.href); e.target.textContent = 'copied'; setTimeout(() => (e.target.textContent = 'copy link'), 1500); } catch { prompt('Copy this URL', location.href); }
  });

  if (auto) doFetch(false);
}
init();
