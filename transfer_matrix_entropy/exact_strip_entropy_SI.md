# Exact orientational entropy on infinite square-lattice strips

## Transfer-matrix definition

A state is a complete open vertical column `C=(r1,...,rW)`. The matrix
element is `T_CC' = exp{x[h(C,C')+(v(C)+v(C'))/2]}`. Its Perron root
gives `beta f=-(ln lambda_max)/W`, and the entropy per particle is
`s/kB=[ln lambda_max-x d(ln lambda_max)/dx]/W`. The derivative in every
reported value was evaluated as `l^T(dT/dx)r/(l^T r)`; centered finite
differences were used only as validation.

For the L particle, the stated rule (left E patch, right W patch) makes
`h(C,C')` directed in the column labels, so `T` is not entrywise symmetric
even after the symmetric split of the vertical energy. Left and right
Perron vectors are therefore both retained. This is the physical
left-to-right compatibility specified in the problem.

## Numerical results for W=3 and W=4

The closure entropies are 0 (avgE), ln(2)=0.693147181 (stick isolated-Z),
and ln(4)=1.386294361 (L-shaped isolated-Z). All quantities below are per
particle; the error columns are absolute errors.

### Stick, W=3

| x | exact s/kB | avgE error | isolated-Z error |
|---:|---:|---:|---:|
| 0.0 | 0.693147181 | 0.693147181 | 0.000000000 |
| 0.5 | 0.670761735 | 0.670761735 | 0.022385446 |
| 1.0 | 0.563935227 | 0.563935227 | 0.129211953 |
| 1.5 | 0.318134360 | 0.318134360 | 0.375012820 |
| 2.0 | 0.127878356 | 0.127878356 | 0.565268824 |
| 3.0 | 0.019618483 | 0.019618483 | 0.673528697 |
| 4.0 | 0.003157181 | 0.003157181 | 0.689989999 |
| 5.0 | 0.000507666 | 0.000507666 | 0.692639515 |
| 6.0 | 0.000080359 | 0.000080359 | 0.693066821 |

### Stick, W=4

| x | exact s/kB | avgE error | isolated-Z error |
|---:|---:|---:|---:|
| 0.0 | 0.693147181 | 0.693147181 | 0.000000000 |
| 0.5 | 0.672657003 | 0.672657003 | 0.020490177 |
| 1.0 | 0.583359310 | 0.583359310 | 0.109787870 |
| 1.5 | 0.344076658 | 0.344076658 | 0.349070522 |
| 2.0 | 0.133017412 | 0.133017412 | 0.560129769 |
| 3.0 | 0.019776235 | 0.019776235 | 0.673370945 |
| 4.0 | 0.003164905 | 0.003164905 | 0.689982275 |
| 5.0 | 0.000508096 | 0.000508096 | 0.692639084 |
| 6.0 | 0.000080384 | 0.000080384 | 0.693066797 |

### L-shaped, W=3

| x | exact s/kB | avgE error | isolated-Z error |
|---:|---:|---:|---:|
| 0.0 | 1.386294361 | 1.386294361 | 0.000000000 |
| 0.5 | 1.368442521 | 1.368442521 | 0.017851840 |
| 1.0 | 1.319013142 | 1.319013142 | 0.067281219 |
| 1.5 | 1.247935623 | 1.247935623 | 0.138358738 |
| 2.0 | 1.166079181 | 1.166079181 | 0.220215180 |
| 3.0 | 1.000771341 | 1.000771341 | 0.385523020 |
| 4.0 | 0.857463565 | 0.857463565 | 0.528830796 |
| 5.0 | 0.744026506 | 0.744026506 | 0.642267855 |
| 6.0 | 0.658733566 | 0.658733566 | 0.727560795 |

### L-shaped, W=4

| x | exact s/kB | avgE error | isolated-Z error |
|---:|---:|---:|---:|
| 0.0 | 1.386294361 | 1.386294361 | 0.000000000 |
| 0.5 | 1.368823267 | 1.368823267 | 0.017471094 |
| 1.0 | 1.317283830 | 1.317283830 | 0.069010531 |
| 1.5 | 1.233333895 | 1.233333895 | 0.152960466 |
| 2.0 | 1.119930766 | 1.119930766 | 0.266363595 |
| 3.0 | 0.837675746 | 0.837675746 | 0.548618615 |
| 4.0 | 0.565149815 | 0.565149815 | 0.821144546 |
| 5.0 | 0.364312355 | 0.364312355 | 1.021982007 |
| 6.0 | 0.233364576 | 0.233364576 | 1.152929785 |

## Crossovers and strong-coupling limits

Equal closure error occurs at `s=(ln q)/2`. The exact results are:

| model | W | crossover x | ground bond slope mu/column | rho(A0) | s(infinity)/kB |
|:--|--:|--:|--:|--:|--:|
| stick | 1 | 1.225043686957 | 1.000000 | 1.000000000 | 0.000000000 |
| stick | 2 | 1.367491430401 | 2.000000 | 1.000000000 | 0.000000000 |
| stick | 3 | 1.445817867180 | 3.000000 | 1.000000000 | 0.000000000 |
| stick | 4 | 1.495597765340 | 4.000000 | 1.000000000 | 0.000000000 |
| L-shaped | 1 | infinity | 0.500000 | 2.000000000 | 0.693147181 |
| L-shaped | 2 | 3.168499787586 | 2.000000 | 1.000000000 | 0.000000000 |
| L-shaped | 3 | 5.559039618969 | 2.500000 | 4.000000000 | 0.462098120 |
| L-shaped | 4 | 3.504893859342 | 4.000000 | 1.000000000 | 0.000000000 |

## Width comparison at finite coupling

| model | x | W=1 | W=2 | W=3 | W=4 |
|:--|--:|--:|--:|--:|--:|
| stick | 1.0 | 0.449623910 | 0.525429303 | 0.563935227 | 0.583359310 |
| stick | 2.0 | 0.105752567 | 0.119807051 | 0.127878356 | 0.133017412 |
| stick | 4.0 | 0.003097949 | 0.003141767 | 0.003157181 | 0.003164905 |
| stick | 6.0 | 0.000080164 | 0.000080310 | 0.000080359 | 0.000080384 |
| L-shaped | 1.0 | 1.355994499 | 1.296998066 | 1.319013142 | 1.317283830 |
| L-shaped | 2.0 | 1.275350289 | 1.041344992 | 1.166079181 | 1.119930766 |
| L-shaped | 4.0 | 1.058481036 | 0.496249092 | 0.857463565 | 0.565149815 |
| L-shaped | 6.0 | 0.884012152 | 0.216713443 | 0.658733566 | 0.233364576 |

## SI-ready interpretation

The transfer result has the required noninteracting limit
`s(0)/kB=ln q` for every width. Increasing directional coupling removes
orientational entropy monotonically. For sticks the all-H strip is the
unique maximally bonded transfer state, so the strong-coupling residual
entropy is zero for W=1-4.

For L particles, every site carries one horizontal and one vertical
patch. Horizontal dimers contribute at most W/2 bonds per added column,
while a maximum matching of the open vertical path contributes
`floor(W/2)`. Thus the ground-state bond slope is
`mu=W/2+floor(W/2)`. Even widths have a unique vertical perfect-matching
pattern and only a finite set of horizontal dimer phases, giving zero
entropy density. Odd widths retain local freedom in the unpaired
vertical patch: rho(A0)=2 for W=1 and 4 for W=3, hence residual entropies
ln(2) and ln(4)/3 per particle, respectively. More generally, the number
of maximum vertical-patch strings is W+1 for odd W, so
`s(infinity)/kB=ln(W+1)/W` for odd W and zero for even W. The W=1-4
sequence is therefore not monotone at strong coupling, although both the
odd-width envelope and the even-width subsequence converge to zero as W
increases. At finite coupling, sticks show rapid convergence with width;
the L curves show much larger even/odd differences once strong local
bonding sets in.

The exact entropy always lies between the two closures. avgE is closer
above the reported crossover, whereas isolated-Z is closer below it.
For L-shaped W=1, the exact entropy approaches (ln 4)/2 only as
x -> infinity, so there is no finite crossover.

## Reproducibility files

- `exact_strip_entropy.py`: construction, analytic derivatives, tests, tables, and plots
- `data/entropy_table_W3_W4.csv`: requested table including lambda derivative and errors
- `data/entropy_curves_W1_W4.csv`: plotted W=1-4 curves
- `data/width_comparison_W1_W4.csv`: selected finite-x width comparison
- `data/crossovers_W1_W4.csv`: equal-error crossover values
- `data/strong_coupling_W1_W4.csv`: ground-state slopes and residual entropies
- `data/validation_summary.txt`: bond-count, normalization, derivative, and direct partition checks
- `figures/exact_strip_entropy.pdf` and `.png`: publication-quality figure
