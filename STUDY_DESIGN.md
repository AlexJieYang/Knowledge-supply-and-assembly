# Study design and operational definitions

We distinguish cited knowledge supplying a reference context from the current publications assembling that context. The pipeline measures reference co-use, matched source and assembler roles, national networks and temporal distributions.

## Populations and attribution

- **Core records:** the existing journal/conference corpus through 2024. Unknown years retain a separate status; future-year records are excluded.
- **Classified background:** 1990-2024 records with country and field information, without a reference minimum. This population supports `full_core_rca.csv`.
- **Primary population P\*:** focal papers from 1990-2024 with country and field information and at least two valid core references jointly classified on both dimensions. C/F/H, primary role accounts and primary RCA use P\*.
- **Benchmark subset:** P\* papers whose required leave-paper-out pools meet the minimum remaining mass. `H_model`, `H0` and `F_model` share this subset.
- **Optional matched semantic subset:** sampled papers with enough references jointly classified by country, field and valid embedding; q is recomputed on that same set.

The induced citation graph retains historical sources without a lower year cutoff, removes duplicate pairs, self-links, invalid dates and future-year references, and allows same-year references. Network summaries use seven natural five-year windows from 1990-1994 through 2020-2024.

Countries describe observed institutional locations. Supplied country weights are validated and small rounding differences corrected. Field vectors use distinct labels at the configured level, with equal-distinct or supplied weights. Their outer product defines country-field attribution. The empirical article uses one broad field label per classified record; the implementation also supports multiple labels.

## Reference contexts

For usable reference j, a_j is its normalised country vector, t_j its field vector and v_j their outer product. For distinct j and k:

```text
r(j, k) = (1 - a_j . a_k) (1 - t_j . t_k)
sigma_ij = sum[k != j] r(j, k) / (m_i - 1)
q_i = sum[j] sigma_ij / m_i
b_ij = sigma_ij / m_i
sum[j] b_ij = q_i
```

m_i is the number of jointly classified references of focal paper i. Geographic co-use g and disciplinary co-use d average the respective factors over identical distinct pairs. Measures are undefined when m_i < 2.

With A = sum(a_j), T = sum(t_j) and V = sum(v_j), the equivalent aggregate expression is:

```text
(m_i - 1) sigma_ij = (m_i - 1)
                     - a_j . (A - a_j)
                     - t_j . (T - t_j)
                     + v_j . (V - v_j)
```

Self-pairs are subtracted explicitly. Sigma is stored as float32; paper q is its mean to retain matched credit. Aggregation uses float64 and numerical tolerances.

Pair contexts are q, g-q, d-q and 1-g-d+q: both boundaries, geography only, discipline only and neither. Coupling is q-g*d within each paper, averaged afterwards. Primary-field sensitivity retains identical reference identities and requires primary-field information for every usable reference. These measures describe provenance, with semantic breadth and scientific outcomes treated as separate quantities.

## Matched role accounts

For country-field unit u, source weight is v_ju and focal-paper weight v_iu:

```text
E_src,u = sum[i,j] v_ju / m_i
B_src,u = sum[i,j] v_ju * b_ij
E_asm,u = sum[i] v_iu
B_asm,u = sum[i] v_iu * q_i
R_role,u = B_role,u / E_role,u
LQ_role,u = R_role,u / world_mean(q)
```

Each exposure account sums to eligible-paper count and each credit account to total q. Zero-exposure rates are undefined; positive exposure with zero credit gives R=0. LQ requires positive world mean q. National rates sum credits and exposures over fields before division.

Source accounts include historical publications according to their use in the current citing window; assembler accounts locate current focal papers. Primary RCA compares a country's field share of assembler exposure with the field's world share within P\*. Classified-background RCA is exported separately.

## Networks and communities

For source country a and receiver country b:

```text
F_ab = sum[i] a_ib * sum[j] a_ja / m_i
H_ab = sum[i] a_ib * sum[j] sigma_ij * a_ja / m_i
sum(F) = N
sum(H) = sum(q)
0 <= H_ab <= F_ab
```

Each paper contributes one unit to F and q to H. H reweights exposure by reference context. A domestic source-receiver link can receive contextual credit through the source's other reference companions.

Collaboration C divides one unit across k(k-1)/2 unordered country pairs when k>=2. Cached C is symmetric, so sum(C)/2 equals the international-paper count. Single-country papers remain in P\*, F/H and assembler accounts.

Domestic diagonals remain in national accounts and are removed for international communities and participation. Weighted Leiden uses an undirected configuration objective for C and directed objectives for F/H. The primary resolution is 1, with sensitivities 0.5, 1.5 and 2 and ten optimisation runs per layer/window. The best objective is retained; repeated-run agreement describes optimisation stability.

Cross-layer AMI and normalised variation of information use common assigned countries. AMI gives countries equal weight; correspondence matrices and alluvial displays use common F exposure. Incoming participation describes source communities used by a receiver; outgoing participation describes receiver communities using a source country's work. F/H comparisons share the current F partition; temporal positions use the fixed 2010-2014 F partition.

Fixed high-exposure sets use baseline international F incoming-plus-outgoing strength, with top-10%, top-5% and top-20% definitions. Within-set shares count international links whose two endpoints belong to the fixed set.

## Reference comparisons

The workflow distinguishes observed citation exposure F, the composition benchmark H0 and empirical contextual reweighting H_context.

Benchmark pools use citing year, receiving field, source type and age bands of 0, 1-2, 3-5, 6-10, 11-20, 21-40 and 41+ years. Types are journal, conference and unknown. Slots enter with focal-paper field weights without 1/m weighting. A source can recur through citations by different papers.

All focal-paper slots are removed before calculating pool means of country, field and joint profiles. For a draw K from a remaining pool:

```text
E[r(j, K)] = 1 - a_j . mean(a_K) - t_j . mean(t_K)
              + v_j . mean(v_K)
```

Expectations are mixed according to the other reference slots' age/type composition. Every required remaining pool must meet `min_null_pool`. Observed and benchmark totals share identical supported papers. The benchmark preserves slot composition while leaving individual cited-work degrees and reference-identity uniqueness unconstrained. H_context instead averages observed sigma in full strata without focal-paper removal.

## Distributions concentration and change

Primary summaries use all eligible papers. Plotting records use deterministic uniform bottom-k hashes. Population distributions combine a zero atom with positive bins; approximate quantiles use upper-bin edges.

Concentration retains the union of units active in either role in any window, including period-specific zeros. Top shares rank each distribution separately. Source-cohort and age intensities divide contextual credit by paper-normalised exposure.

For baseline 2010-2014 and comparison 2020-2024, with exposure shares p and assembler rates R:

```text
within = sum[common u] ((p_u0 + p_u1) / 2) * (R_u1 - R_u0)
composition = sum[common u] ((R_u0 + R_u1) / 2) * (p_u1 - p_u0)
entry = sum[new u] p_u1 * R_u1
exit = -sum[exiting u] p_u0 * R_u0
```

The terms sum to the global change in mean q. Country and field summaries aggregate identical unit terms. Fixed-support means require annual exposure above the threshold from `fixed_support_start` onward and use baseline exposure weights. Endpoint summaries retain 2020-2023 and 2020-2024 separately.

## Classification completion

For M valid core references and m jointly classified references, define K=M(M-1)/2, k=m(m-1)/2 and O=k*q:

```text
lower = O / K
upper = (O + K - k) / K
```

Allow remaining pairs to range from zero to one, then average paper-specific limits. These conditional bounds concern observed core references within P\* and are separate from sampling confidence intervals.

## Conditional models and figures

Paper models use one citing paper per record, field-year and publication-source fixed effects and source-clustered covariance. Predictors are international authorship, log author count, log usable references, mean reference age and field count when identified. Field count is constant under a one-label classification. Complete-case counts and sampling weights are exported.

The country-field-year model relates assembler intensity to log(1+RCA) and log(1+assembler exposure), weights by exposure, includes country-field and field-year fixed effects, and clusters by country and year. The optional AI-profile model uses valid scores in a separate classified subset.

Raincloud components use a common sample. Positive ridge densities are scaled within periods and zero mass is separate. Response surfaces connect observed bin means, masking insufficient support. Role maps share colour scales, undefined cells are grey, and untruncated numerical values remain available. Synthetic figures carry explicit labels.
