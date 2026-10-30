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
