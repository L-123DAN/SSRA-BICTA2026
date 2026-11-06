"""Independent rank/tie AUROC verification of 120 retained WinCLIP+ cells."""
from pathlib import Path
import csv,math,json
BASE=Path(__file__).resolve().parent
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def auc(items):
    pairs=sorted((float(r['pred_score']),int(r['label'])) for r in items);np=sum(p[1] for p in pairs);nn=len(pairs)-np;assert np and nn
    total=0.;i=0
    while i<len(pairs):
        j=i+1
        while j<len(pairs) and pairs[j][0]==pairs[i][0]:j+=1
        total+=sum(p[1] for p in pairs[i:j])*(i+1+j)/2;i=j
    return 100*(total-np*(np+1)/2)/(np*nn)
def main():
    raw=rows(BASE/'sources/results/baselines/baseline_raw.csv');assert len(raw)==120
    for r in raw:
        p=BASE/'winclip_image_scores'/r['category']/('shot_'+r['shot'])/('support_'+r['support_seed'])/'per_image.csv'
        data=rows(p);assert len(data)==int(r['test_count'])
        assert math.isclose(auc(data),float(r['I_AUROC']),abs_tol=1e-8)
    print(json.dumps({'status':'PASS','WinCLIP_image_AUROC_cells':120,'method':'independent rank statistic with average ranks for exact ties','pixel_map_recomputation':False}))
if __name__=='__main__':main()
