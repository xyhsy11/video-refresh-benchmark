"""Three concise static scientific figures; no outcome selection."""
from pathlib import Path
import json
import numpy as np


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out); s=json.loads((out/'summary.json').read_text()); tm=json.loads((out/'timing_summary.json').read_text())
    metrics={r['model']:r for r in s['metrics']}; views=['S','H','HG','HGH']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
    a=axes[0]; values=[metrics['task_ridge_'+v]['value'] for v in views]
    a.bar(views,values,color=['#9babc0','#5479a5','#238a8d','#bb6759']); a.axhline(0,color='black',lw=.7)
    a.axhline(.01,color='#777777',ls=':',lw=1); a.set_ylabel('Nats per mean single intervention'); a.set_title('OOF task value relative to Periodic')
    a=axes[1]; a.bar(views,[metrics['task_ridge_'+v]['regret'] for v in views],color='#5479a5')
    a.axhline(s['periodic_regret'],color='#bb6759',ls='--',label='Periodic regret'); a.set_title('Regret versus matched-tail oracle'); a.set_ylabel('Nats'); a.legend(fontsize=9)
    primary=next(r for r in s['comparisons'] if r['family']=='ridge' and r['contrast']=='HGH_minus_HG')
    fig.suptitle(f"DEVELOPMENT / EXPLORATORY — ridge Delta_HOG = {primary['value_difference']:+.5f} nats",fontsize=12)
    fig.savefig(out/'figure_1_task_value.png'); fig.savefig(out/'figure_1_task_value.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained'); xx=np.arange(4)
    axes[0].plot(xx,[metrics['drift_ridge_'+v]['Spearman'] for v in views],marker='o',label='Cosine drift')
    axes[0].plot(xx,[metrics['task_ridge_'+v]['Spearman'] for v in views],marker='s',label='Signed executable D_P')
    axes[0].set_xticks(xx,views); axes[0].set_ylabel('Weighted OOF Spearman'); axes[0].axhline(0,lw=.6,color='black'); axes[0].legend(); axes[0].set_title('Ranking drift is a separate task')
    axes[1].bar(views,values,color='#238a8d'); axes[1].axhline(0,lw=.7,color='black'); axes[1].axhline(.01,color='#777',ls=':')
    axes[1].set_title('Task value requires useful signed decisions'); axes[1].set_ylabel('Nats per mean single intervention')
    fig.suptitle('DEVELOPMENT / EXPLORATORY — identical clips, folds, targets per task and weights',fontsize=12)
    fig.savefig(out/'figure_2_drift_task.png'); fig.savefig(out/'figure_2_drift_task.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
    a=axes[0]; a.bar(list('ABCD'),[tm[k+'_median_ms']['median'] for k in 'ABCD'],color='#5479a5')
    a.set_ylabel('CPU milliseconds per clip'); a.set_title('Fresh decode + observation/scoring arms')
    a=axes[1]; keys=['D_minus_A_median_ms','D_minus_C_median_ms']; labels=['Total added D-A','HOG increment D-C']
    a.scatter([0,1],[tm[k]['median'] for k in keys],s=60,label='Median',color='#238a8d',zorder=3)
    a.scatter([0,1],[tm[k]['p95'] for k in keys],s=55,marker='x',label='p95',color='#bb6759',zorder=3)
    a.axhline(10,color='#777',ls=':',label='10ms total-added reference'); a.axhline(0,color='black',lw=.6)
    a.set_xticks([0,1],labels); a.set_ylabel('Paired CPU milliseconds'); a.set_title('Within-clip paired repetition differences'); a.legend(fontsize=8)
    fig.suptitle('DEVELOPMENT / EXPLORATORY — 60 components; expensive encoder absent',fontsize=12)
    fig.savefig(out/'figure_3_acquisition_cost.png'); fig.savefig(out/'figure_3_acquisition_cost.pdf'); plt.close(fig)
