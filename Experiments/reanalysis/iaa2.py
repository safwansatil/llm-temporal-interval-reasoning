import csv, os, itertools, collections, random
E='/home/claude/llm-temporal-interval-reasoning/Experiments'
D=os.path.join(E,'exports/stem_relations')
P=os.path.join(E,'results/parsed')
MODELS=['deepseek','gpt-luna','gemini','qwen']
pf=[('stem_50','stem_50_batch_198'),('stem_54','stem_54_batch_263'),
 ('stem_62','stem_62_batch_229'),('stem_63','stem_63_batch_230'),('stem_64','stem_64_batch_241'),
 ('stem_70','stem_70_batch_232'),('stem_102','stem_102_batch_270'),('stem_118','stem_118_batch_271')]
def load(stem):
    rows=list(csv.DictReader(open(os.path.join(D,stem+'_relations.csv'),encoding='utf-8')))
    span={}; rel={}
    for r in rows:
        span[r['from_event']]=r['text_i'].strip(); span[r['to_event']]=r['text_j'].strip()
        rel[(r['from_event'],r['to_event'])]=r['relation_name'].split()[0]
    return span,rel
def load_pred(model,stem):
    f=os.path.join(P,model,stem+'.csv')
    if not os.path.exists(f): return {}
    return {(r['from_event'],r['to_event']):r['pred_symbol'] for r in csv.DictReader(open(f,encoding='utf-8')) if r['pred_symbol']}
def kappa(pairs):
    if not pairs: return float('nan'),0
    N=len(pairs); po=sum(u==v for u,v in pairs)/N
    ma=collections.Counter(u for u,_ in pairs); mb=collections.Counter(v for _,v in pairs)
    pe=sum(ma[l]*mb[l] for l in set(ma)|set(mb))/N**2
    return ((po-pe)/(1-pe) if pe<1 else float('nan')), N

# collect the shared-pair index: (stem, idA, jdA) -> gold label
shared_idx=[]; human=[]
for a,b in pf:
    sa,ra=load(a); sb,rb=load(b)
    inv_a={}; inv_b={}
    for k,v in sa.items(): inv_a.setdefault(v,k)
    for k,v in sb.items(): inv_b.setdefault(v,k)
    for x,y in itertools.combinations([v for v in inv_a if v in inv_b],2):
        ia,ja=inv_a[x],inv_a[y]; ib,jb=inv_b[x],inv_b[y]
        if (ia,ja) in ra and (ib,jb) in rb: key=(ia,ja); u,v=ra[key],rb[(ib,jb)]
        elif (ja,ia) in ra and (jb,ib) in rb: key=(ja,ia); u,v=ra[key],rb[(jb,ib)]
        else: continue
        shared_idx.append((a,key,u)); human.append((u,v))
k_h,N=kappa(human)
acc_h=sum(u==v for u,v in human)/N
print(f"Second annotation pass vs primary gold:  acc={acc_h:.3f}  kappa={k_h:.3f}  (N={N} pairs, 8 stems)\n")
print(f"{'model':<10}{'answered':>10}{'acc':>8}{'kappa':>8}   (same 190 pairs, answered cells only)")
for m in MODELS:
    preds={}
    for a,_ in pf: preds[a]=load_pred(m,a)
    rows=[(g,preds[s].get(k)) for s,k,g in shared_idx if preds[s].get(k)]
    kk,n=kappa(rows); acc=sum(u==v for u,v in rows)/max(n,1)
    print(f"{m:<10}{n:>10}{acc:>8.3f}{kk:>8.3f}")

# cluster bootstrap over the 8 stems for the human kappa
by_stem=collections.defaultdict(list)
for (s,k,g),(u,v) in zip(shared_idx,human): by_stem[s].append((u,v))
stems=list(by_stem); rnd=random.Random(20261006); est=[]
for _ in range(2000):
    samp=[]
    for _ in stems: samp+=by_stem[rnd.choice(stems)]
    kk,_n=kappa(samp)
    if kk==kk: est.append(kk)
est.sort()
print(f"\nhuman kappa 95% CI (cluster bootstrap over 8 stems): [{est[int(.025*len(est))]:.3f}, {est[int(.975*len(est))]:.3f}]")
