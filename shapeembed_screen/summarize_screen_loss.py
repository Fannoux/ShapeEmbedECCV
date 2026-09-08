import os, glob, csv, re
import sys
import pandas as pd

if sys.argv[1]:
    BASE = sys.argv[1]
else:
    BASE = os.environ.get('OUT_BASE', 'shapeembed_beta_screen_out2')
print('config\tlatent\tnorm\tbeta\trecon\tkl\ttotal_val\tclassification')
cols = ['config', 'latent', 'norm', 'beta', 'reconstruction', 'KL', 'total_val']
res_summary = pd.DataFrame(columns=cols)
res_clas_final = pd.DataFrame()
clas_cols=['set', 'scale?', 'clas_method', '', '', '']
n=0
for d in sorted(glob.glob(os.path.join(BASE, '*'))):
    if not os.path.isdir(d):  	continue
    #print(d)
    tag = os.path.basename(d)
    m = re.match(r'ls(\d+)_b([0-9.e+-]+)_lr[0-9.e+-]+_(\w+)_e\d+', tag)
    L, B, NORM = (m.group(1), m.group(2), m.group(3)) if m else ('', '', '')
    last = {}
    p = f'{d}/loss_record.csv'
    if os.path.exists(p):
        r = list(csv.DictReader(open(p)));  last = r[-1] if r else {}
        clf = ''
        rp = f'{d}/run_report.txt'
        if os.path.exists(rp):
            #clf = '\n'.join(
            clf = [l.strip() for l in open(rp) if re.search(r'f1|kappa|accuracy', l, re.I)]
            #)
            #print(tag,'\n', clf)
            res_clas = pd.Series(clf).str.split('[-()]', regex=True , expand=True)
            res_clas['config'] = tag
            res_clas_final = pd.concat([res_clas_final, res_clas], axis=0)
    #print('\t'.join(str(x) for x in
    #    [tag, L, NORM, B, last.get('recon',''), last.get('kl',''), last.get('total_val',''), clf]))
        res_summary.loc[n] = [tag, L, NORM, B, last.get('recon',''), last.get('kl',''), last.get('total_val', '')]
    n+=1
print(res_clas_final.sort_values(3).drop([5], axis=1).to_markdown())
res_clas_final.drop([5], axis=1).to_csv(os.path.join(BASE, 'summary_classification.csv'), index=False)
print()
print(res_summary.sort_values('reconstruction').to_markdown())
res_summary.to_csv(os.path.join(BASE, 'summary_reconstruction.csv'), index=False)
