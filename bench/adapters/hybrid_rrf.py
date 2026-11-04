# -*- coding: utf-8 -*-
"""Hybrid lexical + dense Reciprocal Rank Fusion — whole-note BM25 + multilingual-e5-base without reranker.

Why it is on the board: `bm25` wins on exact entity names but scores 0.00 on zero-shared-word
`paraphrase` queries; `sqlite-graph-memory-vector` finds those paraphrases with `e5-base` but its
cross-encoder reranker scores them below 0.0 and discards every paraphrase answer (100% false
abstention on `paraphrase` and `paraphrase_bridge`). This reference row fuses `bm25` (k1=1.5,
b=0.75) and dense `intfloat/multilingual-e5-base` (`brain_ask.load_encoder("cpu")`, `query_sims`,
`retrieve_candidates`) directly via `sqlite_graph_memory.ppr.rrf_fuse` with no graph hop and no
cross-encoder reranker, using the shared `_extract.best_sentence` extractive floor.
"""
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extract import best_sentence  # noqa: E402
from sqlite_graph_memory import brain_ask as ba, ppr  # noqa: E402

SYSTEMS = {'hybrid-rrf': {}}
K1, B = 1.5, 0.75
CAP_SEQ_RX = re.compile(r"\b[A-Z][a-zA-Z'-]*(?:\s+[A-Z][a-zA-Z'-]*)*")
QWORDS = {
    'In', 'What', 'Which', 'Who', 'When', 'Where', 'Why', 'How',
    'Under', 'On', 'At', 'To', 'From', 'By', 'For',
}


class Adapter:
    def __init__(self, topk=None):
        self.raw = {}
        self._topk = topk

    def info(self):
        return {
            'version': 'reference',
            'url': 'https://huggingface.co/intfloat/multilingual-e5-base',
            'config': {
                'k1': K1,
                'b': B,
                'unit': 'whole note (BM25) + chunked note (e5-base)',
                'embedder': ba.E5_MODEL,
                'topk_retrieve': self._topk or ba.TOPK_RETRIEVE,
                'fusion': 'sqlite_graph_memory.ppr.rrf_fuse (k=%d)' % ppr.RRF_K,
                'reranker': None,
                'graph_hops': 0,
                'device': 'cpu',
                'answer': 'extractive: best-overlap sentence of top-3 notes',
                'abstain': 'capitalized entity word or multi-word proper name absent from vault, or no overlapping sentence',
            },
        }

    def setup(self, vault_dir, work_dir):
        self.raw = {p.stem.lower(): p.read_text(encoding='utf-8') for p in sorted(Path(vault_dir).glob('*.md'))}
        self.tf = {k: Counter(re.findall(r'[a-z0-9]+', v.lower())) for k, v in self.raw.items()}
        self.len = {k: sum(c.values()) for k, c in self.tf.items()}
        self.avg = sum(self.len.values()) / max(len(self.len), 1)
        df = Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(self.tf)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.lower_blob = '\n'.join(self.raw.values()).lower()

        idx = Path(work_dir) / 'index'
        env = dict(os.environ, BRAIN_INDEX_DIR=str(idx), CUDA_VISIBLE_DEVICES='',
                   PYTHONPATH=str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
        subprocess.run([sys.executable, '-m', 'sqlite_graph_memory.index_notes', str(vault_dir)],
                       check=True, env=env, stdout=subprocess.DEVNULL)
        self.ba = ba
        self.emb, self.meta = ba.load_index(idx / '_brain_e5.npy', idx / '_brain_e5_meta.pkl')
        self.enc = ba.load_encoder('cpu')

    def _has_unknown_entity(self, question):
        for m in CAP_SEQ_RX.finditer(question):
            words = [w for w in m.group(0).split() if w not in QWORDS]
            if not words:
                continue
            phrase = re.sub(r"'s$", '', ' '.join(words).lower())
            if len(words) >= 2 and phrase not in self.lower_blob:
                return True
            for w in words:
                clean = re.sub(r"'s$", '', w.lower())
                if clean not in self.lower_blob:
                    return True
        return False

    def ask(self, question):
        topk = self._topk or self.ba.TOPK_RETRIEVE
        terms = re.findall(r'[a-z0-9]+', question.lower())
        scores = []
        for k, c in self.tf.items():
            s = 0.0
            for t in terms:
                f = c.get(t)
                if f:
                    s += self.idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * self.len[k] / self.avg))
            scores.append((s, k))
        scores.sort(key=lambda x: (-x[0], x[1]))
        bm25_ranked = [k for s, k in scores if s > 0][:topk]

        sims = self.ba.query_sims(self.enc, self.emb, question)
        cand = self.ba.retrieve_candidates(sims, self.meta, topk=topk)
        dense_ranked, seen = [], set()
        for i in cand:
            nid = Path(self.meta[i]['path']).stem.lower()
            if nid not in seen:
                seen.add(nid)
                dense_ranked.append(nid)

        ranked = ppr.rrf_fuse(bm25_ranked, dense_ranked)[:20]
        top = float(sims[cand[0]]) if cand else 0.0
        nid, sent, _ = best_sentence(question, ranked, self.raw.get)
        if self._has_unknown_entity(question) or not sent:
            return {'retrieved': ranked, 'answer': None, 'citations': [], 'top_score': round(top, 4)}
        return {
            'retrieved': ranked,
            'answer': sent,
            'citations': [{'note': nid, 'quote': sent}],
            'top_score': round(top, 4),
        }


def make(**kwargs):
    return Adapter(**kwargs)
