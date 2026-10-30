## Day 1 audit
- Data: 307,511 applicants, 8.07% default. Accuracy is a poor metric, so use AUC, log-loss, Brier score and calibration.
- DAYS_EMPLOYED = 365243 in 55,374 rows (18%), a placeholder. Default rate 5.4% vs 8.7%, so replace with missing and add an is_placeholder flag.
- 41 columns over 50% missing, mostly housing details in _AVG/_MODE/_MEDI triplets. Decision pending.
- Bureau has several rows per applicant (mean 5.6, max 116). Aggregate to one row per applicant before joining.
- Class weighting leaves AUC unchanged (0.7535 vs 0.7540) but inflates mean predicted risk to 42% against a true 8% (ECE 0.340, Brier 0.202).
- Platt scaling on the calibration set restores it (ECE 0.0030, Brier 0.0682). Platt intercept -2.432 matches log(n0/n1) = log(11.39) = 2.43. Derive by hand.
- AMT_CREDIT and AMT_GOODS_PRICE correlation 0.988, coefficients +1.27 and -1.11 are an artefact of collinearity.
- X'X rank 183 of 197 (14 exactly redundant directions), so Newton needs a ridge term.
- Subgroups (gender, pensioner, no-bureau): observed vs predicted within about 1.3 standard errors. AUC lower for pensioners and no-bureau applicants (0.72 to 0.74).
- - Redundant columns: 12 relations from the 13 one-hot groups (no constant column in X) plus 2 constant numeric columns (FLAG_MOBIL, FLAG_DOCUMENT_12) = 14. Newton needs a ridge term, lambda = 1/C.
- Gender ablation: AUC falls about 0.003 (LR 0.7540 to 0.7512, XGB 0.7613 to 0.7587). Without gender, predicted default for women is 7.5% vs observed 7.0%, and for men 9.0% vs 10.2% (about 4.5 and 5.7 standard errors).
- AMT_CREDIT coefficient 1.272 with AMT_GOODS_PRICE, 0.233 without; AUC cost 0.003.
- SHAP and built-in importance agree on EXT_SOURCE_2/3, then diverge. Aggregate by original variable next.

1. Why accuracy fails here

About 92 of every 100 applicants repay. Suppose a “model” says “no default” for everyone. It is right 92% of the time and has never caught a single defaulter, so it’s useless to a lender. A high accuracy number can hide a model that does nothing.

That is why this project measures other things: AUC (does it rank risky people above safe ones), log-loss and Brier score (are the probabilities good), and calibration (does “10% risk” mean about 10 in 100).

2. The 365243 value

DAYS_EMPLOYED counts days relative to the application date, so normal values are negative. For example, -2000 means they started their job about 5.5 years earlier. A value of 365243 is positive and equals roughly 1,000 years, which can’t be real.

This is a placeholder, a code the data collectors typed in when there was no real employment length to record. Rows like that usually belong to people with no current job, such as pensioners or unemployed applicants. The default rate in Cell 5 will confirm this: if the rate for those rows differs from everyone else, the placeholder carries real information and you shouldn’t just delete it.

The usual fix has two steps. Replace 365243 with a missing value, then add a new column that records “this person had the placeholder.” That way you fix the broken number without throwing away the signal.

3. Joining the bureau table

Cell 6 shows that each applicant can have many rows, one per past credit account. If you join that directly onto the application table, every applicant is copied once per credit account. The data would then contain duplicates, and some people would count many times over.

The fix is to collapse the bureau table to one row per applicant first. For example, count their past loans, take the average and maximum amounts, and count the overdue ones. Then you join one row to one row.
