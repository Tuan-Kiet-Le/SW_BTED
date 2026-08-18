# Clean embedding evaluation — canonical 138 pairs

Protocol: 5-fold StratifiedKFold, shuffle=True, random_state=42; threshold grid 0.005; train-fold-only threshold selection.

| Model | Mean F1 | Std | Precision | Recall | ROC-AUC | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SBERT_MiniLM | 0.9867 | 0.0267 | 0.9750 | 1.0000 | 1.0000 | 38 | 1 | 99 | 0 |

**SBERT_MiniLM fold thresholds:** 0.6, 0.6, 0.6, 0.6, 0.595
Errors: 107 SP26SE068–SU26SE063 (score=0.597802, label=0, pred=1)

| BGE_Small_v1.5 | 0.9882 | 0.0235 | 0.9778 | 1.0000 | 1.0000 | 38 | 1 | 99 | 0 |

**BGE_Small_v1.5 fold thresholds:** 0.77, 0.75, 0.77, 0.77, 0.77
Errors: 84 SU26SE087–SP26SE001 (score=0.765042, label=0, pred=1)

| MPNet_Base_v2 | 0.9882 | 0.0235 | 0.9778 | 1.0000 | 1.0000 | 38 | 1 | 99 | 0 |

**MPNet_Base_v2 fold thresholds:** 0.665, 0.64, 0.665, 0.665, 0.665
Errors: 84 SU26SE087–SP26SE001 (score=0.661988, label=0, pred=1)
