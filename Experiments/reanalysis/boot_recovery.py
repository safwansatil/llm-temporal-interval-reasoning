import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/work'); sys.path.insert(0,'/home/claude/llm-temporal-interval-reasoning/Experiments/scripts')
import common as C, analysis_lib as L
import recover as R
RES=C.RESULTS
items={it.item_id:it for it in C.load_items()}
man=C.read_json(RES/'run_manifest.json')
items={k:v for k,v in items.items() if k in man['items'] and v.is_primary}
MODELS=['deepseek','gpt-luna','gemini','qwen']
store={}
stems=[]; gold=[]; base=[]
for mid in MODELS: store[mid]=[]
for iid,it in items.items():
    n=it.n; iu,ju=np.triu_indices(n,1)
    gold.append(it.gold[iu,ju]); stems.append(np.full(len(iu), it.stem_id))
    base.append(np.full(len(iu), 1))   # always '<' = code 1
    for mid in MODELS:
        rec=C.read_json(RES/'raw'/mid/f'{iid}.json')
        M2,_=R.recover_matrix(rec['response'].get('content'), it.event_ids, True, True)
        M3,_=R.converse_backfill(M2)
        store[mid].append(M3[iu,ju])
gold=np.concatenate(gold); stems=np.concatenate(stems); base=np.concatenate(base)
for mid in MODELS: store[mid]=np.concatenate(store[mid])
us=np.unique(stems); idx={s:np.where(stems==s)[0] for s in us}
B=1000; rng=np.random.default_rng(20261006)
print(f"{'model':<10}{'acc':>8}{'95% CI':>20}{'vs always-<':>14}{'CI of diff':>22}{'p':>8}")
for mid in MODELS:
    p=store[mid]; corr=(p==gold); bcorr=(base==gold)
    acc=corr.mean(); diff=acc-bcorr.mean()
    A=[];Dd=[]
    for _ in range(B):
        pick=rng.choice(us,size=len(us),replace=True)
        ii=np.concatenate([idx[s] for s in pick])
        A.append(corr[ii].mean()); Dd.append(corr[ii].mean()-bcorr[ii].mean())
    A=np.sort(A); Dd=np.sort(Dd)
    pv=2*min((np.array(Dd)<=0).mean(),(np.array(Dd)>=0).mean())
    print(f"{mid:<10}{acc:>8.3f}  [{A[25]:.3f}, {A[975]:.3f}]  {diff:>+13.3f}   [{Dd[25]:+.3f}, {Dd[975]:+.3f}]{pv:>8.3f}")
print(f"\nalways-'<' accuracy = {(base==gold).mean():.4f}")
