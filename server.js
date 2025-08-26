// server.js (CommonJS)
// Minimal RAG with Ollama embeddings + TinyLlama generation

require('dotenv').config();

const express = require('express');
const axios = require('axios');
const fs = require('fs');


const OLLAMA = process.env.OLLAMA_HOST?.startsWith('http') ? process.env.OLLAMA_HOST : `http://${process.env.OLLAMA_HOST}`;
const GEN_MODEL = process.env.GEN_MODEL || 'tinyllama';
const EMBED_MODEL = process.env.EMBED_MODEL || 'nomic-embed-text';
const PORT = process.env.PORT || 5000;
const KB_PATH = './kb.json';

const app = express();
app.use(express.json({ limit: '10mb' }));

let kb = []; // { id, text, embedding: number[], timestamp }

function saveKB() {
  fs.writeFileSync(KB_PATH, JSON.stringify(kb, null, 2));
}

function loadKB() {
  if (fs.existsSync(KB_PATH)) {
    kb = JSON.parse(fs.readFileSync(KB_PATH, 'utf8'));
  } else {
    kb = [
      {
        id: 'd1',
        text: 'TinyLlama is a small language model optimized for fast inference on modest hardware.',
        timestamp: new Date().toISOString()
      },
      {
        id: 'd2',
        text: 'React Native builds mobile apps using JavaScript and native widgets for iOS and Android.',
        timestamp: new Date().toISOString()
      }
    ];
  }
}

async function embedOne(text) {
  const r = await axios.post(`${OLLAMA}/api/embeddings`, {
    model: EMBED_MODEL,
    input: text
  });
  return r.data.embedding;
}

async function ensureEmbeddings() {
  const toEmbed = kb.filter(d => !d.embedding);
  for (const d of toEmbed) {
    d.embedding = await embedOne(d.text);
  }
  saveKB();
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    kbSize: kb.length
  });
});

app.get('/kb', (req, res) => {
  res.json(kb.map(({ id, text, timestamp }) => ({ id, text, timestamp })));
});

app.post('/ingest', async (req, res) => {
  try {
    const docs = req.body.documents || [];
    for (const doc of docs) {
      if (!doc.id || !doc.text) continue;
      const embedding = await embedOne(doc.text);
      kb.push({
        id: doc.id,
        text: doc.text,
        embedding,
        timestamp: new Date().toISOString()
      });
    }
    saveKB();
    res.json({ added: docs.length });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

app.post('/chat', async (req, res) => {
  try {
    const { query, topK = 3 } = req.body;
    if (!query || !query.trim()) return res.status(400).json({ error: 'query is required' });

    const qemb = await embedOne(query);
    const ranked = kb
      .map(d => ({ ...d, score: cosine(qemb, d.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    const context = ranked.map(d => `[${d.id}] ${d.text}`).join('\n---\n');

    const prompt =
`You are a helpful assistant. Use the CONTEXT to answer clearly.
If the answer is not in the context, say you are not sure.

CONTEXT:
${context}

USER: ${query}
ASSISTANT:`;

    const t0 = Date.now();
    const r = await axios.post(`${OLLAMA}/api/generate`, {
      model: GEN_MODEL,
      prompt,
      stream: false
    });
    const elapsed = (Date.now() - t0) / 1000;

    res.json({
      response: r.data.response,
      elapsed,
      sources: ranked.map(x => ({
        id: x.id,
        score: +x.score.toFixed(4),
        timestamp: x.timestamp
      }))
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});


app.listen(PORT, async () => {
  loadKB();
  await ensureEmbeddings();
  console.log(`RAG server running on http://127.0.0.1:${PORT}  |  KB chunks: ${kb.length}`);
});
