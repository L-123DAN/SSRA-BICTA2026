"""Offline CPU reproduction from packaged frozen CSVs. No external dataset needed.

Run `python reproduce_statistics.py` beside this file in the extracted handoff.
Read-only verification against delivered statistics. Python >= 3.10, stdlib only.
"""
from pathlib import Path
from collections import defaultdict
import csv,math,statistics as st,random,json
BASE=Path(__file__).resolve().parent
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def rank(xs):
    out=[0.]*len(xs);order=sorted(range(len(xs)),key=lambda i:xs[i]);i=0
    while i<len(xs):
        j=i+1
        while j<len(xs) and xs[order[j]]==xs[order[i]]:j+=1
        for k in order[i:j]:out[k]=(i+1+j)/2
        i=j
    return out
def mean(xs):return sum(xs)/len(xs)
def corr(xs,ys):
    mx,my=mean(xs),mean(ys);num=sum((x-mx)*(y-my) for x,y in zip(xs,ys));return num/math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
def main():
    sources={'PromptAD':BASE/'sources/paper/ssra_ei_draft/data/visa_verified.csv','WinCLIP+':BASE/'sources/results/baselines/baseline_raw.csv','BTAD':BASE/'sources/results/csv/ssra_btad_raw.csv'}
    groups=defaultdict(list)
    for d,p in sources.items():
        rows=read(p);assert len(rows)==(45 if d=='BTAD' else 120)
        for r in rows:
            for m in ('I_AUROC','P_AUROC','AUPRO'):groups[d,r['category'],int(r['shot']),m].append(float(r[m]))
    desc={k:{'mean':st.mean(v),'sd':st.stdev(v),'range':max(v)-min(v),'worst':min(v),'best':max(v)} for k,v in groups.items()}
    for r in read(BASE/'statistics/per_category_recomputed.csv'):
        v=desc[r['detector'],r['category'],int(r['shot']),r['metric']]
        assert all(math.isclose(float(r[f]),v[f],abs_tol=1e-10) for f in v)
    cats=sorted({k[1] for k in desc if k[0]=='PromptAD' and k[2]==1})
    for r in read(BASE/'statistics/threshold_sensitivity.csv'):
        sd,rg=float(r['sd_threshold_pp']),float(r['range_threshold_pp']);p=[desc['PromptAD',c,1,'I_AUROC']['sd']>=sd or desc['PromptAD',c,1,'I_AUROC']['range']>=rg for c in cats];w=[desc['WinCLIP+',c,1,'I_AUROC']['sd']>=sd or desc['WinCLIP+',c,1,'I_AUROC']['range']>=rg for c in cats]
        a=sum(x and y for x,y in zip(p,w));b=sum(x and not y for x,y in zip(p,w));c=sum(not x and y for x,y in zip(p,w));d=sum(not x and not y for x,y in zip(p,w));po=(a+d)/12;pe=((a+b)*(a+c)+(c+d)*(b+d))/144;k=(po-pe)/(1-pe)
        assert [a,b,c,d]==[int(r[x]) for x in ('both_sensitive','promptad_only','winclip_only','both_insensitive')]
        assert math.isclose(float(r['kappa']),k,abs_tol=1e-12)
    effects=read(BASE/'statistics/category_effects.csv')
    for r in effects:
        c,m=r['category'],r['metric'];scope,con=r['scope'],r['contrast']
        if scope=='cross_1shot':v=desc['WinCLIP+',c,1,m]['sd']-desc['PromptAD',c,1,m]['sd']
        else:
            det=scope.removesuffix('_shot');a,b=desc[det,c,1,m],desc[det,c,4,m]
            v={'sd_1_minus_4':a['sd']-b['sd'],'range_1_minus_4':a['range']-b['range'],'worst_4_minus_1':b['worst']-a['worst'],'mean_4_minus_1':b['mean']-a['mean']}[con]
        assert math.isclose(float(r['value_pp']),v,abs_tol=1e-10)
    for r in read(BASE/'statistics/leave_one_category_out.csv'):
        vs=[float(e['value_pp']) for e in effects if e['scope']==r['scope'] and e['metric']==r['metric'] and e['contrast']==r['contrast'] and e['category']!=r['omitted_category']]
        assert math.isclose(float(r['mean_pp']),st.mean(vs),abs_tol=1e-10) and math.isclose(float(r['median_pp']),st.median(vs),abs_tol=1e-10)
    for r in read(BASE/'statistics/bootstrap_recomputed.csv'):
        scope,metric,contrast=r['scope'],r['metric'],r['contrast']
        if scope=='cross_1shot':vs=[desc['WinCLIP+',c,1,metric]['sd' if contrast.startswith('support') else 'mean']-desc['PromptAD',c,1,metric]['sd' if contrast.startswith('support') else 'mean'] for c in cats]
        else:vs=[float(e['value_pp']) for e in effects if e['scope']==scope and e['metric']==metric and e['contrast']==contrast.removesuffix('_pp')]
        fun=st.mean if r['statistic']=='mean' else st.median;rng=random.Random(int(r['seed']));draws=sorted(fun([vs[rng.randrange(len(vs))] for _ in vs]) for _ in range(int(r['draws'])))
        assert math.isclose(draws[249],float(r['lower_pp']),abs_tol=1e-10) and math.isclose(draws[9749],float(r['upper_pp']),abs_tol=1e-10)
    for r in read(BASE/'statistics/mean_dispersion_association.csv'):
        vs=[desc[r['detector'],c,1,'I_AUROC'] for c in cats if c!=r['omitted_category']]
        rho=corr(rank([v['mean'] for v in vs]),rank([v[r['dispersion']] for v in vs]));assert math.isclose(rho,float(r['spearman_rho']),abs_tol=1e-12)
    stages=read(BASE/'sources/paper/ssra_ei_draft/data/stage1_verified.csv')
    assert len(stages)==45
    for r in read(BASE/'statistics/factorial_recomputed.csv'):
        mat={(int(x['support_seed']),int(x['training_seed'])):float(x['I_AUROC']) for x in stages if x['category']==r['category']};assert len(mat)==15
        ss=sorted({k[0] for k in mat});ts=sorted({k[1] for k in mat});sm={s:st.mean(mat[s,t] for t in ts) for s in ss};tm={t:st.mean(mat[s,t] for s in ss) for t in ts};grand=st.mean(mat.values());total=sum((v-grand)**2 for v in mat.values())
        v={'support_marginal_sd_pp':st.stdev(sm.values()),'training_marginal_sd_pp':st.stdev(tm.values()),'Q':st.stdev(sm.values())/(st.stdev(tm.values())+1e-8),'support_share':3*sum((x-grand)**2 for x in sm.values())/total,'training_share':5*sum((x-grand)**2 for x in tm.values())/total,'residual_share':sum((v-sm[s]-tm[t]+grand)**2 for (s,t),v in mat.items())/total}
        assert all(math.isclose(float(r[k]),value,abs_tol=1e-10) for k,value in v.items())
    print(json.dumps({'status':'PASS','configuration_sources':[120,120,45,45],'recomputed':'all per-category, 9 threshold/kappa, effects, all LOO, 14 bootstrap, 12 associations, 3 factorial summaries','GPU_training':False}))
if __name__=='__main__':main()
