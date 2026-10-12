import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,'/home/claude/work')
sys.path.insert(0,'/home/claude/llm-temporal-interval-reasoning/Experiments/scripts')
import common as C, analysis_lib as L, allen as A
import recover as R

RES = C.RESULTS
items = {it.item_id: it for it in C.load_items()}
man = C.read_json(RES/'run_manifest.json')
items = {k:v for k,v in items.items() if k in man['items'] and v.is_primary}
MODELS = ['deepseek','gpt-luna','gemini','qwen','mistral']

def metrics(gold, pred):
    """upper-triangle metrics, missing counted as wrong (same convention as A1)."""
    Cm = L.confusion(gold, pred)
    return L.metrics_from_conf(Cm)

rows=[]
for mid in MODELS:
    acc={}; agg={k:0 for k in ['rows_strict','rows_R1','rows_R2','cells_R2']}
    G=[]; P={'strict':[], 'R1':[], 'R1R2':[], 'R1R2conv':[]}
    for iid,it in items.items():
        rec = C.read_json(RES/'raw'/mid/f'{iid}.json')
        content = rec['response'].get('content')
        n = it.n; iu, ju = np.triu_indices(n,1)
        Mstrict = C.parse_matrix(content, it.event_ids)['matrix']
        M1,_   = R.recover_matrix(content, it.event_ids, use_R1=True,  use_R2=False)
        M2,st  = R.recover_matrix(content, it.event_ids, use_R1=True,  use_R2=True)
        for k in agg: agg[k]+=st[k]
        M3,_f  = R.converse_backfill(M2)
        G.append(it.gold[iu,ju])
        P['strict'].append(Mstrict[iu,ju]); P['R1'].append(M1[iu,ju])
        P['R1R2'].append(M2[iu,ju]);        P['R1R2conv'].append(M3[iu,ju])
    g=np.concatenate(G)
    out={'model':mid}
    for k in P:
        p=np.concatenate(P[k]); m=metrics(g,p)
        out[f'{k}_cov']=float((p!=C.MISSING).mean()); out[f'{k}_acc']=m['accuracy']; out[f'{k}_f1']=m['macro_f1']
    out.update(agg); rows.append(out)
D=pd.DataFrame(rows)
D.to_csv('/home/claude/work/recovery_table.csv', index=False)
pd.set_option('display.width',200)
print(D[['model','strict_cov','R1_cov','R1R2_cov','R1R2conv_cov']].to_string(index=False))
print()
print(D[['model','strict_f1','R1_f1','R1R2_f1','R1R2conv_f1']].to_string(index=False))
print()
print(D[['model','strict_acc','R1_acc','R1R2_acc','R1R2conv_acc']].to_string(index=False))
