import csv, os, itertools, collections
D='/home/claude/llm-temporal-interval-reasoning/Experiments/exports/stem_relations'
pairs_files=[('stem_50','stem_50_batch_198'),('stem_54','stem_54_batch_263'),
 ('stem_62','stem_62_batch_229'),('stem_63','stem_63_batch_230'),('stem_64','stem_64_batch_241'),
 ('stem_70','stem_70_batch_232'),('stem_102','stem_102_batch_270'),('stem_118','stem_118_batch_271')]

def load(stem):
    f=os.path.join(D,stem+'_relations.csv')
    rows=list(csv.DictReader(open(f,encoding='utf-8')))
    span={}; rel={}
    for r in rows:
        span[r['from_event']]=r['text_i'].strip(); span[r['to_event']]=r['text_j'].strip()
        rel[(r['from_event'],r['to_event'])]=r['relation_name'].split()[0]
    return span, rel

tot_shared=0; tot_agree=0; per=[]
conf=collections.Counter()
for a,b in pairs_files:
    sa,ra=load(a); sb,rb=load(b)
    # map span -> id in each
    inv_a={}; inv_b={}
    for k,v in sa.items(): inv_a.setdefault(v,k)
    for k,v in sb.items(): inv_b.setdefault(v,k)
    shared=[v for v in inv_a if v in inv_b]
    n_ag=0; n_sh=0
    for x,y in itertools.combinations(shared,2):
        ia,ja=inv_a[x],inv_a[y]; ib,jb=inv_b[x],inv_b[y]
        if (ia,ja) in ra and (ib,jb) in rb:
            u,v=ra[(ia,ja)],rb[(ib,jb)]
        elif (ja,ia) in ra and (jb,ib) in rb:
            u,v=ra[(ja,ia)],rb[(jb,ib)]
        else: continue
        n_sh+=1; conf[(u,v)]+=1
        if u==v: n_ag+=1
    per.append((a,len(shared),n_sh,n_ag,n_ag/max(n_sh,1)))
    tot_shared+=n_sh; tot_agree+=n_ag

print(f"{'stem':<12}{'shared ev':>10}{'pairs':>8}{'agree':>8}{'rate':>8}")
for s,e,n,ag,r in per: print(f"{s:<12}{e:>10}{n:>8}{ag:>8}{r:>8.3f}")
print(f"{'TOTAL':<12}{'':>10}{tot_shared:>8}{tot_agree:>8}{tot_agree/tot_shared:>8.3f}")

# Cohen kappa
labels=sorted({l for k in conf for l in k})
N=sum(conf.values())
po=tot_agree/N
ma=collections.Counter(); mb=collections.Counter()
for (u,v),c in conf.items(): ma[u]+=c; mb[v]+=c
pe=sum(ma[l]*mb[l] for l in labels)/N**2
print(f"\nCohen kappa = {(po-pe)/(1-pe):.3f}   (po={po:.3f}, pe={pe:.3f}, N={N})")
print("\nTop disagreements (A -> B):")
for (u,v),c in sorted(conf.items(), key=lambda x:-x[1]):
    if u!=v and c>=3: print(f"   {u:>3} vs {v:<3}  {c}")
