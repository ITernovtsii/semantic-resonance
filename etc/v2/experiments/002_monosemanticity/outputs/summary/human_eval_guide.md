# Human Evaluation Guide: Expert Semantic Purity

## Task
For each expert row in the CSV, review the top-50 tokens (ranked by PMI — tokens most *characteristic* of this expert, not just most frequent). Rate the expert's semantic coherence on a 1-10 scale and provide a brief category label.

## Scoring Rubric

| Score | Description | Example |
|-------|-------------|---------|
| **1** | Random garbage — no discernible pattern | "the(2.1); quantum(1.8); ran(1.5); blue(1.3); if(1.2)..." |
| **2-3** | Very weak — maybe 2-3 related words among noise | Mostly random with a couple sports terms |
| **4-5** | Weak pattern — a theme is visible but noisy | Mix of science terms with unrelated words |
| **6-7** | Clear pattern — dominant theme with some outliers | Mostly geography terms, ~20% noise |
| **8-9** | Strong coherence — nearly all tokens share a theme | Almost entirely music-related tokens |
| **10** | Perfect — every token belongs to the same category | All tokens are punctuation, or all are color words |

## Syntactic vs Semantic Flag

After scoring, mark whether the pattern is **syntactic** or **semantic**:

- **Syntactic**: The pattern is about word *type* or *function*, not meaning
  - Examples: "all punctuation", "all prepositions", "all verb suffixes", "all numbers"
  - These can score high (10 = all commas) but aren't *semantically* meaningful

- **Semantic**: The pattern is about word *meaning* or *topic*
  - Examples: "sports terms", "medical vocabulary", "geographic names", "technology"
  - This is what we're primarily interested in

## Important Notes

1. **PMI values in parentheses**: Higher PMI means the token is more *characteristic* of this expert (appears disproportionately here vs. other experts). A common word with high PMI is informative.

2. **BPE artifacts**: Tokens may have `Ġ` prefix (= space before word in GPT-2 BPE). Ignore this prefix when judging meaning. `Ġscience` = " science".

3. **Subword fragments**: Tokens like `tion`, `ment`, `ing` are BPE subwords. An expert specializing in these is *syntactic* (suffix-oriented), not semantic.

4. **Don't over-interpret**: If you have to stretch to find a pattern, score lower. We want honest ratings.

5. **Model identity is blinded**: You don't know which model produced which CSV. Rate purely on token coherence.

## Process

1. Open one CSV file at a time
2. For each row (expert):
   a. Read the `top_50_tokens` column
   b. Score 1-10 in `human_purity_score`
   c. Write a 1-3 word label in `human_category_label` (e.g., "sports", "punctuation", "mixed/unclear")
   d. Mark `human_syntactic_or_semantic` as "syntactic" or "semantic" (or "mixed")
3. Take breaks every ~30 experts to avoid fatigue
4. Expected time: ~45-60 minutes per CSV

## Anchor Examples (calibration)

**Score 10 (semantic)**: `Ġscience; Ġresearch; Ġexperiment; Ġlaboratory; Ġhypothesis; Ġdiscovery; Ġchemistry; Ġbiology; Ġphysics; Ġscientist...`
→ Label: "science/research", Flag: semantic

**Score 10 (syntactic)**: `.; ,; ?; !; :; ;; "; (; ); -...`
→ Label: "punctuation", Flag: syntactic

**Score 7 (semantic)**: `Ġwar; Ġarmy; Ġbattle; Ġsoldier; Ġmilitary; Ġdefeat; Ġcity; Ġempire; Ġking; Ġterritory...`
→ Label: "military/history", Flag: semantic (some noise: city, empire are borderline)

**Score 4 (mixed)**: `Ġtime; Ġmusic; Ġworld; Ġbased; Ġhowever; Ġincluding; Ġpart; Ġrecord; Ġband; Ġarea...`
→ Label: "mixed/unclear", Flag: mixed

**Score 1**: `Ġthe; 23; Ġwas; Ġfield; ution; Ġalso; Ġred; Ġby; 19; Ġwhile...`
→ Label: "random", Flag: mixed
