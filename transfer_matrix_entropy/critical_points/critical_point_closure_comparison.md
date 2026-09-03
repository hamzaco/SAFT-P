# Exact strip comparison at representative critical points

## What is being compared

The original fully occupied strip cannot test the nondirectional
interaction because every nearest-neighbor pair is then occupied. Here
each site can instead be vacant or carry one of q orientations. The
critical-point benchmark uses a periodic transverse boundary (an infinite
cylinder) so that every site retains the bulk coordination number four;
the earlier open-strip entropy calculation is not altered. For
neighboring occupied sites the energy is `-eps_nd`, with an additional
`-eps_d` for a correctly aligned directional bond. The positive logged
`mu` is the manuscript simulation's occupancy cost.

For a column C with particle number n(C), occupied-contact count a_v(C),
and directional-bond count d_v(C), the exact grand-canonical matrix is

`T_CC' = exp{ eps_nd[A_h+(A_v(C)+A_v(C'))/2]
              + eps_d[D_h+(D_v(C)+D_v(C'))/2]
              - mu[n(C)+n(C')]/2 }`.

Finite-width strips have no true phase transition. The tabulated values
are exact strip thermodynamics evaluated at representative 2D critical
triples from the project logs, not newly claimed strip critical points.
The sparse extension reaches eps_d=5.8 using successful points from
the original Monte Carlo critical-line continuation. A lower-coupling
subset was additionally examined in a later finite-size-scaling campaign;
the lack of that same secondary check at every higher-coupling point is
not a failure of the original MC calculation and no point is excluded
from the present transfer-matrix comparison.

The Perron vectors define the exact stationary column Markov chain
`P(C'|C)=T_CC' r_C'/(lambda r_C)`. Writing a new column as its occupancy
pattern O' plus its occupied-site orientations R' gives the exact
nonnegative decomposition `H(C'|C)=H(O'|C)+H(R'|C,O')`. We use
`s_int=H(R'|C,O')/<N'>` as the environment-conditioned internal entropy
per occupied particle. avgE sets this term to zero and isolated-Z sets it
to ln(q). This conditional definition avoids subtracting an unrelated
reference entropy and includes both nondirectional and directional
interactions in the exact Gibbs weights.

## Numerical comparison

| model | W | eps_nd | eps_d | rho | exact s_int | avgE error | isolated-Z error | closer | input status |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|:--|
| stick | 3 | 1.762747 | 0.00 | 0.500000 | 0.693147 | 0.693147 | 0.000000 | isolated-Z | exact anchor |
| stick | 4 | 1.762747 | 0.00 | 0.500000 | 0.693147 | 0.693147 | 0.000000 | isolated-Z | exact anchor |
| stick | 3 | 1.512717 | 1.00 | 0.505287 | 0.608500 | 0.608500 | 0.084647 | isolated-Z | logged MC + FSS-checked |
| stick | 4 | 1.512717 | 1.00 | 0.476790 | 0.619343 | 0.619343 | 0.073804 | isolated-Z | logged MC + FSS-checked |
| stick | 3 | 1.279846 | 1.80 | 0.570119 | 0.357395 | 0.357395 | 0.335752 | isolated-Z | logged MC + FSS-checked |
| stick | 4 | 1.279846 | 1.80 | 0.530377 | 0.368586 | 0.368586 | 0.324561 | isolated-Z | logged MC + FSS-checked |
| stick | 3 | 1.214832 | 1.96 | 0.587703 | 0.299474 | 0.299474 | 0.393673 | avgE | logged MC + FSS-checked |
| stick | 4 | 1.214832 | 1.96 | 0.567617 | 0.302435 | 0.302435 | 0.390712 | avgE | logged MC + FSS-checked |
| stick | 3 | 1.105412 | 2.20 | 0.530425 | 0.224917 | 0.224917 | 0.468231 | avgE | logged MC critical-line point |
| stick | 4 | 1.105412 | 2.20 | 0.484934 | 0.226820 | 0.226820 | 0.466328 | avgE | logged MC critical-line point |
| stick | 3 | 0.907749 | 2.60 | 0.463151 | 0.142512 | 0.142512 | 0.550635 | avgE | logged MC critical-line point |
| stick | 4 | 0.907749 | 2.60 | 0.402499 | 0.154088 | 0.154088 | 0.539059 | avgE | logged MC critical-line point |
| stick | 3 | 0.784389 | 3.00 | 0.551119 | 0.080134 | 0.080134 | 0.613013 | avgE | logged MC critical-line point |
| stick | 4 | 0.784389 | 3.00 | 0.560957 | 0.077001 | 0.077001 | 0.616146 | avgE | logged MC critical-line point |
| stick | 3 | 0.566073 | 4.00 | 0.529095 | 0.024980 | 0.024980 | 0.668167 | avgE | logged MC critical-line point |
| stick | 4 | 0.566073 | 4.00 | 0.539038 | 0.024803 | 0.024803 | 0.668344 | avgE | logged MC critical-line point |
| stick | 3 | 0.505158 | 5.00 | 0.485292 | 0.008494 | 0.008494 | 0.684653 | avgE | logged MC critical-line point |
| stick | 4 | 0.505158 | 5.00 | 0.471766 | 0.008990 | 0.008990 | 0.684157 | avgE | logged MC critical-line point |
| stick | 3 | 0.390833 | 5.80 | 0.514175 | 0.003788 | 0.003788 | 0.689359 | avgE | logged MC critical-line point |
| stick | 4 | 0.390833 | 5.80 | 0.520984 | 0.003757 | 0.003757 | 0.689391 | avgE | logged MC critical-line point |
| L-shaped | 3 | 1.762747 | 0.00 | 0.500000 | 1.386294 | 1.386294 | 0.000000 | isolated-Z | exact anchor |
| L-shaped | 4 | 1.762747 | 0.00 | 0.500000 | 1.386294 | 1.386294 | 0.000000 | isolated-Z | exact anchor |
| L-shaped | 3 | 1.513433 | 1.02 | 0.474265 | 1.330344 | 1.330344 | 0.055950 | isolated-Z | logged MC + FSS-checked |
| L-shaped | 4 | 1.513433 | 1.02 | 0.499476 | 1.320746 | 1.320746 | 0.065548 | isolated-Z | logged MC + FSS-checked |
| L-shaped | 3 | 1.353086 | 1.80 | 0.454826 | 1.230084 | 1.230084 | 0.156210 | isolated-Z | logged MC + FSS-checked |
| L-shaped | 4 | 1.353086 | 1.80 | 0.626358 | 1.185029 | 1.185029 | 0.201266 | isolated-Z | logged MC + FSS-checked |
| L-shaped | 3 | 1.249506 | 2.60 | 0.257007 | 1.092125 | 1.092125 | 0.294170 | isolated-Z | logged MC + FSS-checked |
| L-shaped | 4 | 1.249506 | 2.60 | 0.578399 | 0.997505 | 0.997505 | 0.388789 | isolated-Z | logged MC + FSS-checked |
| L-shaped | 3 | 1.227555 | 2.80 | 0.322291 | 1.056319 | 1.056319 | 0.329975 | isolated-Z | logged MC critical-line point |
| L-shaped | 4 | 1.227555 | 2.80 | 0.784532 | 0.945796 | 0.945796 | 0.440499 | isolated-Z | logged MC critical-line point |
| L-shaped | 3 | 1.181526 | 3.20 | 0.233644 | 0.972084 | 0.972084 | 0.414211 | isolated-Z | logged MC critical-line point |
| L-shaped | 4 | 1.181526 | 3.20 | 0.791079 | 0.847891 | 0.847891 | 0.538403 | isolated-Z | logged MC critical-line point |
| L-shaped | 3 | 1.088309 | 3.66 | 0.162955 | 0.858548 | 0.858548 | 0.527746 | isolated-Z | logged MC critical-line point |
| L-shaped | 4 | 1.088309 | 3.66 | 0.707161 | 0.742976 | 0.742976 | 0.643318 | isolated-Z | logged MC critical-line point |
| L-shaped | 3 | 1.103772 | 4.18 | 0.112921 | 0.739667 | 0.739667 | 0.646628 | isolated-Z | logged MC critical-line point |
| L-shaped | 4 | 1.103772 | 4.18 | 0.817442 | 0.636775 | 0.636775 | 0.749520 | avgE | logged MC critical-line point |
| L-shaped | 3 | 0.969690 | 5.56 | 0.087016 | 0.394814 | 0.394814 | 0.991481 | avgE | logged MC critical-line point |
| L-shaped | 4 | 0.969690 | 5.56 | 0.840242 | 0.430927 | 0.430927 | 0.955368 | avgE | logged MC critical-line point |
| L-shaped | 3 | 0.907398 | 5.80 | 0.156659 | 0.367048 | 0.367048 | 1.019246 | avgE | logged MC critical-line point |
| L-shaped | 4 | 0.907398 | 5.80 | 0.916494 | 0.408252 | 0.408252 | 0.978042 | avgE | logged MC critical-line point |

## Closure switch along the sampled critical line

The following values use piecewise-linear interpolation of eps_nd and
mu between adjacent critical-input triples; they are interpolation
diagnostics, not additional Monte Carlo critical points.

| model | W | eps_d at equal error | status |
|:--|--:|--:|:--|
| stick | 3 | 1.829383 | interpolation between FSS-checked logged points |
| stick | 4 | 1.853161 | interpolation between FSS-checked logged points |
| L-shaped | 3 | 4.359527 | interpolation between logged MC critical-line points |
| L-shaped | 4 | 3.893402 | interpolation between logged MC critical-line points |

## Interpretation

At eps_d=0 the calculation recovers s_int=ln(q) to numerical
precision, so isolated-Z is exact in the genuinely isolated-orientation
limit. Directional bonding lowers the environment-conditioned internal
entropy below ln(q). The table determines, point by point, whether that
loss is large enough to make the zero-entropy avgE limit closer than the
full isolated degeneracy.

For stick, the closure switch occurs in the lower-coupling subset that
also received the secondary FSS check. For L-shaped particles, the
switch occurs at higher coupling and depends more strongly on strip
width; this is a finite-width convergence issue in the exact-strip
benchmark, not evidence that the underlying MC points failed.

This is a controlled exact-strip benchmark of the entropy closure. It
does not replace the full SAFT-P critical-line recomputation because the
finite strip has no critical singularity and the SAFT-P plaquette classes
contain additional coarse-grained variables.
