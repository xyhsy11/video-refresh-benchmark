"""Publication layout pass for Experiment25 figure2; reads estimates only.

This changes presentation, never scientific outcomes, intervals or scales.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/continuation_value_gap_v1'


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    keys=['G_star','G_matched','R_star','G_star_U','G_matched_U','R_star_U']
    fig,axs=plt.subplots(1,2,figsize=(10.4,4.8),layout='constrained',sharey=True)
    for ax,name in zip(axs,['ucf','hmdb']):
        p=OUT/name/'estimates.json'
        if not p.exists():
            ax.text(.5,.5,'HMDB: NOT EVALUATED\nSee frozen population and execution gate',ha='center',va='center',transform=ax.transAxes)
            ax.set_axis_off();continue
        e=json.loads(p.read_text());values=np.array([e['estimates'][k]['mean'] for k in keys])
        ax.bar(range(6),values,color=['#a54b43','#477d98','#b28b3e']*2,width=.7)
        for j,k in enumerate(keys):
            ci=e['estimates'][k]['interval']
            if ci is not None:
                ax.errorbar(j,values[j],yerr=[[values[j]-ci[0]],[ci[1]-values[j]]],fmt='none',color='k',capsize=3,lw=1)
        ax.axhline(0,color='.4',lw=.7);ax.axvline(2.5,color='.7',ls=':',lw=.8)
        ax.set_xticks(range(6),['G*','G matched','R*','G* U','G matched U','R* U'],rotation=30,ha='right')
        ax.set_title(name.upper()+('\nFull-cohort interval unidentifiable: 2 singleton classes' if name=='ucf' else '\nApproximate 95% CR2/Satterthwaite intervals'),fontsize=10,pad=10)
        ax.set_ylabel('Hierarchically weighted CE gain / regret (nats)')
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.margins(y=.18)
        ax.set_xlabel('Periodic-repair tail                Uniform legal tail\nBaseline: Periodic CE                 Baseline: V_U',fontsize=9,labelpad=10)
    fig.suptitle('One privileged intervention at a uniformly chosen reset state\nOracle-selected gain, matched-tail opportunity and regret',fontsize=12)
    for extension in ['pdf','png']:fig.savefig(OUT/f'figure_2_endpoints.{extension}',dpi=240)
    plt.close(fig)


if __name__=='__main__':main()
