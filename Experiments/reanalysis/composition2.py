import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/llm-temporal-interval-reasoning/Experiments/scripts')
import common as C, analysis_lib as L
from allen import COMP_SINGLE, MISSING, full_matrix_from_upper
RES=C.RESULTS
items={it.item_id:it for it in C.load_items()}
man=C.read_json(RES/'run_manifest.json')
items={k:v for k,v in items.items() if k in man['items'] and v.is_primary}
MODELS=['deepseek','gpt-luna','gemini','qwen']
rows=[]
for mid in MODELS:
    ne=nc=ea=ca=0
    for iid,it in items.items():
        n=it.n; idx=np.arange(n); G=it.gold.astype(int)
        M=C.parse_matrix(C.read_json(RES/'raw'/mid/f'{iid}.json')['response'].get('content'),it.event_ids)['matrix']
        U=np.full((n,n),MISSING,dtype=np.int16); iu,ju=np.triu_indices(n,1); U[iu,ju]=M[iu,ju]
        Pm=full_matrix_from_upper(U).astype(int)          # exactly ctx.upper_matrix
        okik=(Pm==G)&(G!=MISSING)
        Gv=np.where(G==MISSING,0,G+6)
        Sg=COMP_SINGLE[Gv[:,:,None],Gv[None,:,:]]
        Sg=np.where((G[:,:,None]!=MISSING)&(G[None,:,:]!=MISSING),Sg,MISSING)
        avail=okik[:,:,None]&okik[None,:,:]&(Sg==G[:,None,:])
        avail[idx,idx,:]=False; avail[:,idx,idx]=False
        comp=avail.any(axis=1)[iu,ju]
        wrong=Pm[iu,ju]!=G[iu,ju]
        ne+=int(wrong.sum());  ea+=int((comp&wrong).sum())
        nc+=int((~wrong).sum()); ca+=int((comp&~wrong).sum())
    rows.append({'model':mid,'errors':ne,'err_comp_available':ea,'share_of_errors':ea/max(ne,1),
                 'correct':nc,'cor_comp_available':ca,'share_of_correct':ca/max(nc,1),
                 'ratio':(ca/max(nc,1))/max(ea/max(ne,1),1e-12)})
D=pd.DataFrame(rows); pd.set_option('display.width',250)
print("=== COMPOSITION AVAILABILITY (your A5 premise definition) + the missing control ===")
print(D.round(4).to_string(index=False))
