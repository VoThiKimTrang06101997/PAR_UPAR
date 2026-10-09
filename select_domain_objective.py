"""Select coupled SupCon/alignment weights from TRUE LODO train-heldout experiments.

All folds were trained from ImageNet, without any full-source teacher. This
selector never uses the official private test labels. Fixed 0.5 thresholds
avoid fitting operating points on held-out labels during the comparison.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report-dir', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--candidates', default='none:0:0,moderate:0.04:0.01,strong:0.08:0.02')
    ap.add_argument('--min-improvement', type=float, default=0.002)
    ap.add_argument('--max-weakest-drop', type=float, default=0.004)
    args = ap.parse_args()
    trials = []
    for item in args.candidates.split(','):
        tag, sc, align = item.split(':')
        trials.append((tag, float(sc), float(align)))
    rows = []
    for tag, sc, align in trials:
        folds = []
        for heldout in (0,1,2):
            p = args.report_dir / f'prototype_lodo_seed{args.seed}_heldout{heldout}_{tag}.json'
            if not p.exists():
                raise FileNotFoundError(f'Missing REAL LODO training report: {p}')
            rec = json.loads(p.read_text(encoding='utf-8'))
            assert rec['heldout_domain'] == heldout
            assert not rec['teacher_used'] and not rec['training_domain_leakage']
            if abs(rec['contrastive_weight'] - sc) > 1e-8 or abs(rec['alignment_weight']-align)>1e-8:
                raise ValueError(f'Unexpected weights in {p}')
            folds.append(float(rec['best_selection']))
        scores = np.asarray(folds, dtype=np.float64)
        rows.append({
            'name':tag, 'contrastive_weight':sc, 'alignment_weight':align,
            'fold_scores':scores.tolist(), 'mean':float(scores.mean()),
            'worst':float(scores.min()), 'std':float(scores.std()),
            'aggregate':float(.65 * scores.mean() + .35 * scores.min() - .05*scores.std()),
        })
    baseline = next((r for r in rows if r['contrastive_weight']==0 and r['alignment_weight']==0), None)
    if baseline is None:
        raise ValueError('A zero regularization baseline is required')
    rows.sort(key=lambda x:x['aggregate'], reverse=True)
    champion = rows[0]
    accepted = (
        champion['name'] != baseline['name']
        and champion['aggregate'] > baseline['aggregate'] + args.min_improvement
        and champion['worst'] >= baseline['worst'] - args.max_weakest_drop
    )
    selected = champion if accepted else baseline
    report = {'selected':selected,'baseline':baseline,'accepted_contrastive':accepted,
              'candidates':rows,
              'notes': 'Strict train-domain LODO folds, ImageNet only, no full-source teacher. '
                       'These are source-domain transfer tests, not the private test.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':main()
