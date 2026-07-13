from __future__ import annotations

# Curated subset of the Loughran-McDonald (2011) finance-specific sentiment
# lexicon — the standard for 10-K/10-Q tone (generic lexicons like TextBlob
# mis-score finance text: "liability", "cost", "capital" aren't negative in
# accounting). The full LM master dictionary has ~2350 negative / ~290
# uncertainty words; embedded here is a high-frequency subset so the module is
# self-contained (no fragile external fetch). The primary text-*change* signal
# needs no word list, so this only affects the secondary tone factors. Swapping
# in the full LM CSV is a documented refinement.

_NEGATIVE = """
loss losses adverse adversely decline declined declines declining decrease
decreased decreases decreasing weak weakness weaknesses weakened difficult
difficulties difficulty litigation damages default defaults deficiency
deficiencies deficit deteriorate deteriorated deteriorating deterioration
discontinued downturn doubt doubtful fail failed failing failure failures
impair impaired impairment impairments insolvency insolvent negative negatively
penalty penalties poor recession restructuring terminate terminated termination
unfavorable warn warned warning worse worsen worsening worst breach claims
complaint complaints concern concerns contraction crisis critical cyclical
disruption disruptions downgrade downgraded drop dropped exposure fraud
fraudulent hurt harmed impacted inability ineffective inefficient insufficient
lawsuit lawsuits layoff layoffs liquidation misconduct obsolete overcapacity
plunged pressure pressures recall recalls reduction reductions shortfall
slowdown slowed sluggish strain stress unable underperform underperformance
unemployment unpaid unprofitable weaker writedown writedowns writeoff writeoffs
against bankruptcy bankruptcies burden challenged challenges closure closures
cancel cancelled cancellation curtail curtailed defective delay delays deny
denied depressed diminished distress erosion forfeit forfeited impede injury
insufficiency lose losing negatives nonperforming problematic prolonged
questioned recessionary shortage shortages shrink shrinking suffered suffering
threat threats troubled uncollectible unsuccessful volatility wrongdoing
""".split()

_UNCERTAINTY = """
approximate approximately assume assumed assumes assuming assumption
assumptions believe believed believes contingency contingencies could depend
depended depending depends fluctuate fluctuated fluctuates fluctuating
fluctuation fluctuations indefinite indefinitely likelihood may maybe might
possibility possibilities possible possibly precaution precautionary predict
predictability predicted predicting prediction predictions probability probable
probably risk risked riskier riskiest risks risky seldom sometimes somewhat
speculate speculative sudden suddenly tentative tentatively uncertain
uncertainties uncertainty unclear undetermined unexpected unexpectedly
unforeseen unknown unpredictability unpredictable unproven unsettled unusual
unusually vague variability variable variables variation variations vary
varying volatile volatilities volatility exposure contingent conditional
""".split()

NEGATIVE_WORDS = frozenset(w.lower() for w in _NEGATIVE)
UNCERTAINTY_WORDS = frozenset(w.lower() for w in _UNCERTAINTY)
