import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/work'); sys.path.insert(0,'/home/claude/llm-temporal-interval-reasoning/Experiments/scripts')
import common as C, analysis_lib as L
import recover as R
RES=C.RESULTS
items={it.item_id:it for it in C.load_items()}
man=C.read_json(RES/'run_manifest.json')
items={k:v for k,v in items.items() if k in man['items'] and v.is_primary}
MODELS=['deepseek','gpt-luna','gemini','qwen']
rows=[]
for mid in MODELS:
    g=[];s=[];r2=[];cv=[]
    for iid,it in items.items():
        rec=C.read_json(RES/'raw'/mid/f'{iid}.json'); content=rec['response'].get('content')
        n=it.n; iu,ju=np.triu_indices(n,1)
        Ms=C.parse_matrix(content,it.event_ids)['matrix']
        M2,_=R.recover_matrix(content,it.event_ids,True,True)
        M3,_=R.converse_backfill(M2)
        G=it.gold[iu,ju]; S=Ms[iu,ju]; A2=M2[iu,ju]; A3=M3[iu,ju]
        g.append(G); s.append(S); r2.append(A2); cv.append(A3)
    g=np.concatenate(g); s=np.concatenate(s); r2=np.concatenate(r2); cv=np.concatenate(cv)
    strict_mask = s!=C.MISSING
    new_r2 = (s==C.MISSING)&(r2!=C.MISSING)
    new_cv = (r2==C.MISSING)&(cv!=C.MISSING)
    rows.append({'model':mid,
        'strict_cells':int(strict_mask.sum()), 'strict_acc':float((s[strict_mask]==g[strict_mask]).mean()),
        'R2_new_cells':int(new_r2.sum()),      'R2_new_acc':float((r2[new_r2]==g[new_r2]).mean()),
        'conv_new_cells':int(new_cv.sum()),    'conv_new_acc':float((cv[new_cv]==g[new_cv]).mean()),
        'agree_on_strict':float((r2[strict_mask]==s[strict_mask]).mean())})
D=pd.DataFrame(rows); pd.set_option('display.width',250)
print(D.to_string(index=False))
print("\nalways-before accuracy on the full upper triangle = 0.6068")
