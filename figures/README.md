# Figures for SCALING_LAW_REPORT.md

| file | produced by | shown as |
|---|---|---|
| `dense-lossvsparams.png` | `plot_isoflop(rows_where(all_rows, 0))` | Fig 1 |
| `lossvscompute.png` | `plot_best_loss(best_loss_vs_compute(all_rows, centres=C_TARGETS))` | Fig 2 |
| `lossvsactiveparams.png` | `plot_loss_vs_N(all_rows)` | Fig 3 |

See §9 of the report for the snippet that regenerates all three. The plot functions
call `plt.show()`, which clears the figure, so `plt.show` must be shadowed before
calling them rather than saving afterwards.
