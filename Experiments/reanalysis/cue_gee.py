import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/llm-temporal-interval-reasoning/Experiments/scripts')
import common as C, analysis_lib as L
import statsmodels.api as sm
RES=C.RESULTS
items={it.item_id:it for it in C.load_items()}
man=C.read_json(RES/'run_manifest.json')
items={k:v for k,v in items.items() if k in man['items'] and v.is_primary}
MODELS=['deepseek','gpt-luna','gemini','qwen']
rec=[]
for iid,it in items.items():
    n=it.n; f=L.item_features(it); iu,ju=np.triu_indices(n,1)
    G=it.gold[iu,ju].astype(int)
    pos_i,pos_j=f.pos[iu],f.pos[ju]; sd=np.abs(f.sent[iu]-f.sent[ju])
    ok=np.isin(G,[1,-1])&~np.isnan(pos_i)&~np.isnan(pos_j)
    first_is_i=G==1
    pf=np.where(first_is_i,pos_i,pos_j); ps=np.where(first_is_i,pos_j,pos_i)
    narr_rev=np.where(ok,(pf>ps).astype(float),np.nan)
    list_contra=(G==-1).astype(float)    # list presents the later event first
    preds={}
    for m in MODELS:
        M=C.parse_matrix(C.read_json(RES/'raw'/m/f'{iid}.json')['response'].get('content'),it.event_ids)['matrix']
        preds[m]=M[iu,ju]
    for m in MODELS:
        d=pd.DataFrame({'stem':it.stem_id,'gold':G,'narr_rev':narr_rev,'list_contra':list_contra,
                        'sd':sd,'pred':preds[m],'ok':ok}); d['model']=m; rec.append(d)
D=pd.concat(rec,ignore_index=True)
D=D[D.ok].copy(); D['answered']=D.pred!=C.MISSING; D['correct']=(D.pred==D.gold).astype(int)
print("=== 2x2 CUE CELLS (answered cells, accuracy) ===")
for m in MODELS:
    d=D[(D.model==m)&D.answered]
    t=d.groupby(['list_contra','narr_rev']).correct.agg(['mean','size'])
    print(f"\n{m}")
    for (lc,nr),r in t.iterrows():
        tag=f"list {'CONTRA' if lc else 'agree '} / narration {'REVERSED' if nr else 'forward '}"
        print(f"   {tag}  acc={r['mean']:.3f}  n={int(r['size']):,}")
print("\n=== GEE logistic, correct ~ list_contra * narr_rev + sent_dist, clustered by passage ===")
print("(gold relation is OMITTED: for before/after pairs it is collinear with list_contra by construction)")
print(f"\n{'model':<10}{'interaction OR':>16}{'95% CI':>22}{'p':>9}")
for m in MODELS:
    d=D[(D.model==m)&D.answered].dropna(subset=['narr_rev'])
    X=pd.DataFrame({'const':1.0,'list_contra':d.list_contra.values,'narr_rev':d.narr_rev.values,
                    'inter':d.list_contra.values*d.narr_rev.values,'sd':d.sd.values})
    try:
        r=sm.GEE(d.correct.values,X,groups=d.stem.values,family=sm.families.Binomial(),
                 cov_struct=sm.cov_struct.Independence()).fit()
        b=r.params['inter']; ci=r.conf_int().loc['inter']; p=r.pvalues['inter']
        print(f"{m:<10}{np.exp(b):>16.3f}   [{np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}]{p:>9.4f}")
    except Exception as e:
        print(f"{m:<10}  failed: {e}")
