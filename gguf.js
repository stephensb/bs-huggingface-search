// gguf.js — read a GGUF file header over HTTP range requests and compute the exact parameter count
// from the tensor dimensions. Works in the browser (CORS is open on the Hub's CDN) and in Node.
(function (root) {
  const SCALAR_SIZE = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
  // general.file_type values (llama.cpp LlamaFileType)
  const FILE_TYPES = { 0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 7: 'Q8_0', 8: 'Q5_0', 9: 'Q5_1', 10: 'Q2_K', 11: 'Q3_K_S', 12: 'Q3_K_M',
    13: 'Q3_K_L', 14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S', 17: 'Q5_K_M', 18: 'Q6_K', 19: 'IQ2_XXS', 20: 'IQ2_XS', 21: 'Q2_K_S', 22: 'IQ3_XS',
    23: 'IQ3_XXS', 24: 'IQ1_S', 25: 'IQ4_NL', 26: 'IQ3_S', 27: 'IQ3_M', 28: 'IQ2_S', 29: 'IQ2_M', 30: 'IQ4_XS', 31: 'IQ1_M', 32: 'BF16',
    36: 'TQ1_0', 37: 'TQ2_0', 38: 'MXFP4' };
  class NeedMore extends Error {}

  function parse(buffer) {
    const dv = new DataView(buffer);
    const dec = new TextDecoder();
    let pos = 0;
    const need = (n) => { if (pos + n > buffer.byteLength) throw new NeedMore(); };
    const u32 = () => { need(4); const v = dv.getUint32(pos, true); pos += 4; return v; };
    const u64 = () => { need(8); const v = Number(dv.getBigUint64(pos, true)); pos += 8; return v; };
    const str = (decode) => { const n = u64(); need(n); const s = decode ? dec.decode(new Uint8Array(buffer, pos, n)) : null; pos += n; return s; };
    const scalar = (t) => {
      need(SCALAR_SIZE[t]);
      let v;
      switch (t) {
        case 0: v = dv.getUint8(pos); break; case 1: v = dv.getInt8(pos); break;
        case 2: v = dv.getUint16(pos, true); break; case 3: v = dv.getInt16(pos, true); break;
        case 4: v = dv.getUint32(pos, true); break; case 5: v = dv.getInt32(pos, true); break;
        case 6: v = dv.getFloat32(pos, true); break; case 7: v = dv.getUint8(pos) !== 0; break;
        case 10: v = Number(dv.getBigUint64(pos, true)); break; case 11: v = Number(dv.getBigInt64(pos, true)); break;
        case 12: v = dv.getFloat64(pos, true); break;
        default: throw new Error('bad GGUF value type ' + t);
      }
      pos += SCALAR_SIZE[t];
      return v;
    };
    const value = (t, decode) => {
      if (t === 8) return str(decode);
      if (t === 9) {
        const et = u32(); const n = u64();
        if (et === 8) { for (let i = 0; i < n; i++) str(false); return `[${n} strings]`; }
        if (et === 9) { for (let i = 0; i < n; i++) value(9, false); return `[${n} arrays]`; }
        if (!SCALAR_SIZE[et]) throw new Error('bad GGUF array type ' + et);
        need(n * SCALAR_SIZE[et]); pos += n * SCALAR_SIZE[et];
        return `[${n} values]`;
      }
      return scalar(t);
    };

    if (u32() !== 0x46554747) throw new Error('not a GGUF file');
    const version = u32();
    if (version < 2 || version > 3) throw new Error('unsupported GGUF version ' + version);
    const nTensors = u64();
    const nKv = u64();
    const kv = {};
    for (let i = 0; i < nKv; i++) {
      const k = str(true);
      const t = u32();
      kv[k] = value(t, !k.startsWith('tokenizer.'));
    }
    let params = 0;
    let maxDims = 0;
    for (let i = 0; i < nTensors; i++) {
      str(false);
      const nd = u32();
      let p = 1;
      for (let d = 0; d < nd; d++) p *= u64();
      u32(); u64();
      params += p;
      if (nd > maxDims) maxDims = nd;
    }
    const arch = kv['general.architecture'];
    return {
      version, nTensors, kv, params, headerBytes: pos, arch,
      name: kv['general.name'], sizeLabel: kv['general.size_label'],
      fileType: FILE_TYPES[kv['general.file_type']] || (kv['general.file_type'] != null ? 'type ' + kv['general.file_type'] : undefined),
      contextLength: arch ? kv[arch + '.context_length'] : undefined,
      blockCount: arch ? kv[arch + '.block_count'] : undefined,
      expertCount: arch ? kv[arch + '.expert_count'] : undefined,
    };
  }

  async function fetchRange(url, start, end, signal) {
    const r = await fetch(url, { headers: { Range: `bytes=${start}-${end}` }, signal });
    if (r.status !== 206) {
      if (r.status === 200) { r.body && r.body.cancel && r.body.cancel(); throw new Error('server ignored the byte range request'); }
      throw new Error('HTTP ' + r.status + (r.status === 401 || r.status === 403 ? ' (gated or private repo)' : ''));
    }
    return new Uint8Array(await r.arrayBuffer());
  }

  // Reads just enough of the file to parse the header. Big-vocab models need ~10 MB because the
  // tokenizer vocabulary lives in the header, so start with a 4 MB chunk and double on demand.
  async function readHeader(url, { chunk = 4 << 20, max = 64 << 20, signal } = {}) {
    let buf = new Uint8Array(0);
    let size = chunk;
    for (;;) {
      const part = await fetchRange(url, buf.length, buf.length + size - 1, signal);
      const nb = new Uint8Array(buf.length + part.length);
      nb.set(buf); nb.set(part, buf.length);
      buf = nb;
      try {
        return { ...parse(buf.buffer), bytesRead: buf.length };
      } catch (e) {
        if (!(e instanceof NeedMore)) throw e;
        if (part.length < size) throw new Error('file ended inside the header');
        if (buf.length >= max) throw new Error('header larger than ' + (max >> 20) + ' MB');
        size = Math.min(size * 2, max - buf.length);
      }
    }
  }

  root.GGUF = { readHeader, parse, FILE_TYPES };
})(typeof globalThis !== 'undefined' ? globalThis : window);
