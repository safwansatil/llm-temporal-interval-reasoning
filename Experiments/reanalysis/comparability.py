import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/llm-temporal-interval-reasoning/Experiments/scripts')
import common as C, analysis_lib as L
RES=C.RESULTS
items={it.item_id:it for it in C.load_items()}
man=C.read_json(RES/'run_manifest.json')
items={k:v for k,v in items.items() if k in man['items'] and v.is_primary}
MODELS=['deepseek','gpt-luna','gemini','qwen']
G=[];S=[];pred={m:[] for m in MODELS}
for iid,it in items.items():
    n=it.n; iu,ju=np.triu_indices(n,1)
    G.append(it.gold[iu,ju]); S.append(np.full(len(iu),it.stem_id))
    for m in MODELS:
        M=C.parse_matrix(C.read_json(RES/'raw'/m/f'{iid}.json')['response'].get('content'),it.event_ids)['matrix']
        pred[m].append(M[iu,ju])
g=np.concatenate(G); st=np.concatenate(S)
for m in MODELS: pred[m]=np.concatenate(pred[m])
ans={m:pred[m]!=C.MISSING for m in MODELS}
common=np.logical_and.reduce([ans[m] for m in MODELS])
print(f"=== COMMON-SUBSET TABLE: cells answered by all 4 models: {common.sum():,} of {len(g):,} ({common.mean():.1%}) ===")
print(f"{'model':<10}{'acc':>8}{'macroF1':>9}   (own answered cells, for reference)")
for m in MODELS:
    mm=L.metrics_simple(g[common],pred[m][common],L.LABELS)
    own=L.metrics_simple(g[ans[m]],pred[m][ans[m]],L.LABELS)
    print(f"{m:<10}{mm['accuracy']:>8.3f}{mm['macro_f1']:>9.3f}   {own['accuracy']:.3f} / {own['macro_f1']:.3f}")
ab=L.metrics_simple(g[common],np.full(common.sum(),1),L.LABELS)
print(f"{'always <':<10}{ab['accuracy']:>8.3f}{ab['macro_f1']:>9.3f}")

print("\n=== PAIR CONCENTRATION ===")
ns=np.array([it.n for it in items.values()]); npairs=ns*(ns-1)//2
o=np.argsort(-npairs); cum=np.cumsum(npairs[o])/npairs.sum()
for k in [0.05,0.10,0.25]:
    kk=int(round(k*len(ns))); print(f"  top {k:.0%} of passages ({kk:3d}) hold {cum[kk-1]:.1%} of all pairs")
print(f"  passages with >40 events: {(ns>40).sum()} ({(ns>40).mean():.1%}) hold {npairs[ns>40].sum()/npairs.sum():.1%} of all pairs")
print(f"  total pairs = {npairs.sum():,}")

print("\n=== COVERAGE: cell-weighted vs passage-macro-averaged ===")
print(f"{'model':<10}{'cell-wtd':>10}{'passage-avg':>13}")
for m in MODELS:
    per=[(pred[m][st==s]!=C.MISSING).mean() for s in np.unique(st)]
    print(f"{m:<10}{(pred[m]!=C.MISSING).mean():>10.3f}{np.mean(per):>13.3f}")

print("\n=== MACRO-P / MACRO-R (missing = FN) ===")
print(f"{'model':<10}{'macroP':>9}{'macroR':>9}{'macroF1':>9}")
for m in MODELS:
    Cm=L.confusion(g,pred[m]); mm=L.metrics_from_conf(Cm)
    tp=np.array([Cm[k,k] for k in range(13)],float)
    predtot=Cm[:,:13].sum(0).astype(float); goldtot=Cm.sum(1).astype(float)
    P=np.divide(tp,predtot,out=np.zeros(13),where=predtot>0)
    Rr=np.divide(tp,goldtot,out=np.zeros(13),where=goldtot>0)
    print(f"{m:<10}{P.mean():>9.3f}{Rr.mean():>9.3f}{mm['macro_f1']:>9.3f}")

print("\n=== SUPPORT PER RELATION: pairs vs PASSAGES ===")
sym=C.CODE2SYM if hasattr(C,'CODE2SYM') else None
print(f"{'rel':>4}{'pairs':>9}{'passages':>10}")
for c in L.LABELS:
    msk=g==c; ps=len(np.unique(st[msk])) if msk.any() else 0
    nm=sym[c] if sym else str(c)
    print(f"{nm:>4}{int(msk.sum()):>9}{ps:>10}")
