import sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/llm-temporal-interval-reasoning/Experiments/scripts')
import common as C, analysis_lib as L
RES=C.RESULTS
items={it.item_id:it for it in C.load_items()}
man=C.read_json(RES/'run_manifest.json')
items={k:v for k,v in items.items() if k in man['items'] and v.is_primary}
MODELS=['deepseek','gpt-luna','gemini','qwen']
CACHE=RES/'analysis'/'cache'/'repair'
rows=[]
for mid in MODELS:
  for var in ['i','ii']:
    g=[];raw=[];rep=[]
    for iid,it in items.items():
        n=it.n; iu,ju=np.triu_indices(n,1)
        M=C.parse_matrix(C.read_json(RES/'raw'/mid/f'{iid}.json')['response'].get('content'),it.event_ids)['matrix']
        f=CACHE/f'{mid}__{iid}__{var}.json'
        U=np.array(json.load(open(f))['upper'],dtype=np.int16) if f.exists() else None
        g.append(it.gold[iu,ju]); raw.append(M[iu,ju]); rep.append(U if U is not None and U.shape[0]==len(iu) else M[iu,ju])
    g=np.concatenate(g);raw=np.concatenate(raw);rep=np.concatenate(rep)
    ansm=raw!=C.MISSING
    # control: fill every missing cell with '<' (code 1)
    ctrl=raw.copy(); ctrl[~ansm]=1
    changed=ansm&(rep!=raw)
    wr=(raw[changed]!=g[changed])&(rep[changed]==g[changed])   # wrong -> right
    rw=(raw[changed]==g[changed])&(rep[changed]!=g[changed])   # right -> wrong
    rows.append({'model':mid,'variant':var,
      'raw_acc':(raw==g).mean(),'repaired_acc':(rep==g).mean(),'fill_with_<_acc':(ctrl==g).mean(),
      'raw_f1':L.metrics_from_conf(L.confusion(g,raw))['macro_f1'],
      'repaired_f1':L.metrics_from_conf(L.confusion(g,rep))['macro_f1'],
      'fill_f1':L.metrics_from_conf(L.confusion(g,ctrl))['macro_f1'],
      'answered_changed':int(changed.sum()),'edits_wrong_to_right':int(wr.sum()),'edits_right_to_wrong':int(rw.sum()),
      'edit_precision':float(wr.sum()/max(changed.sum(),1))})
D=pd.DataFrame(rows); pd.set_option('display.width',260)
print("=== REPAIR vs TRIVIAL FILL CONTROL (accuracy, all pairs) ===")
print(D[['model','variant','raw_acc','repaired_acc','fill_with_<_acc']].round(4).to_string(index=False))
print("\n=== macro-F1 ===")
print(D[['model','variant','raw_f1','repaired_f1','fill_f1']].round(4).to_string(index=False))
print("\n=== PRECISION OF THE REPAIR'S EDITS TO ANSWERED CELLS ===")
print(D[['model','variant','answered_changed','edits_wrong_to_right','edits_right_to_wrong','edit_precision']].round(3).to_string(index=False))
