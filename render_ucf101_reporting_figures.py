"""Post-exposure layout correction only; scientific inputs are hash-locked unchanged."""
from ucf101_proxy_common import OUT, seal_check


def main():
    seal_check(True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import pandas as pd
    rows = pd.read_csv(OUT/'secondary_intervals.csv', float_precision='round_trip')
    rows = rows[(rows.budget == 4) & (rows.pool == 'C')].set_index('contrast')
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 160})
    fig, ax = plt.subplots(figsize=(9.8, 4.0), layout='constrained')
    for i, name in enumerate(['directional_g', 'KL_g', 'g']):
        r = rows.loc[name]
        ax.errorbar(r['mean'], i,
            xerr=[[r['mean']-r.CI95_lower], [r.CI95_upper-r['mean']]], fmt='o', capsize=5)
    ax.axvline(0, color='grey', lw=1)
    ax.set_yticks(range(3), ['Signed directional difference', 'KL difference', 'Total g'])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set(xlabel='CE contribution difference, nats',
        title='Exact descriptive decomposition: g = directional difference + KL difference\nComponent intervals are marginal/unadjusted; no causal attribution')
    for ext in ['png', 'pdf']:
        fig.savefig(OUT/f'figures/decomposition.{ext}')
    plt.close(fig)


if __name__ == '__main__':
    main()
