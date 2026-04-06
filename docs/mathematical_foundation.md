# Mathematical Foundation

## 1. Daily universe

Let:

- $M_d$ = maximum valid files for day $d$
- $p_e$ = global error percentage

Then:

- Global error universe: $E_d = M_d \cdot p_e$
- Valid unique universe: $C_d = M_d - E_d$

## 2. Hierarchical allocation

Errors are not allocated directly from the whole daily universe into every subtype.
The correct order is:

1. daily maximum,
2. global error,
3. error family,
4. error subtype.

This macro-to-micro hierarchy avoids logical inconsistencies.

## 3. Why combinations and permutations are not enough

Classical combinatorics is useful for counting arrangements or constructing nonces, but it is not enough to model the whole synthetic universe because:

- the universe is not equiprobable,
- categories have weighted frequencies,
- error cases are hierarchical,
- operational realism demands expected frequencies, not just possible arrangements.

## 4. Why discrete probability distributions fit better

The simulator models a weighted categorical universe. For that reason, normalized weights and integer allocation methods are more appropriate than pure combinatorial reasoning.

## 5. Integer preservation

The largest remainder method is used to preserve exact totals after converting probabilistic weights into integer counts.
