#!/usr/bin/env python3
"""
Check language disparities in MIMIC cardiac cohort
Compare language gaps vs racial gaps
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
data_path = Path(__file__).parent.parent / "FIDES" / "results" / "mimic_full" / "mimic_cardiac_fides_full.csv"

if not data_path.exists():
    print(f"Data not found at {data_path}")
    exit(1)

df = pd.read_csv(data_path)

# Create mortality column (invert good_outcome)
df['mortality'] = 1 - df['good_outcome']

print("=" * 80)
print("DISPARITIES ANALYSIS: Race vs Language")
print("=" * 80)

overall_mortality = df['mortality'].mean()
print(f"\nOverall mortality rate: {overall_mortality * 100:.1f}%")

# ============================================================================
# RACIAL DISPARITIES
# ============================================================================
print("\n" + "=" * 80)
print("RACIAL DISPARITIES")
print("=" * 80)

race_mortality = df.groupby('race').agg({
    'mortality': ['sum', 'count', 'mean']
}).round(4)
race_mortality.columns = ['deaths', 'n', 'mortality_rate']
race_mortality['gap_pp'] = (race_mortality['mortality_rate'] - overall_mortality) * 100
race_mortality = race_mortality.sort_values('gap_pp', ascending=False)

print("\n" + race_mortality.to_string())
print(f"\nLargest racial gap: {race_mortality['gap_pp'].abs().max():.1f} pp")
print(f"Mean absolute racial gap: {race_mortality['gap_pp'].abs().mean():.1f} pp")

# ============================================================================
# LANGUAGE DISPARITIES
# ============================================================================
print("\n" + "=" * 80)
print("LANGUAGE DISPARITIES (RAW)")
print("=" * 80)

# Check what language values exist
print(f"\nUnique languages in dataset: {df['language'].nunique()}")
print("\nTop 15 languages:")
lang_counts = df['language'].value_counts().head(15)
print(lang_counts)

language_mortality = df.groupby('language').agg({
    'mortality': ['sum', 'count', 'mean']
}).round(4)
language_mortality.columns = ['deaths', 'n', 'mortality_rate']
language_mortality['gap_pp'] = (language_mortality['mortality_rate'] - overall_mortality) * 100
language_mortality = language_mortality.sort_values('gap_pp', ascending=False)

print("\n" + language_mortality.to_string())

print(f"\nLargest language gap: {language_mortality['gap_pp'].abs().max():.1f} pp")
print(f"Mean absolute language gap: {language_mortality['gap_pp'].abs().mean():.1f} pp")
print(f"Number of languages with >1.0 pp gap: {len(language_mortality[language_mortality['gap_pp'].abs() > 1.0])}")

# ============================================================================
# LANGUAGE DISPARITIES (BUCKETED)
# ============================================================================
print("\n" + "=" * 80)
print("LANGUAGE DISPARITIES (BUCKETED: English vs Non-English)")
print("=" * 80)

df['language_bucketed'] = df['language'].apply(lambda x: 'English' if x == 'English' else 'Non-English')

lang_buck_mortality = df.groupby('language_bucketed').agg({
    'mortality': ['sum', 'count', 'mean']
}).round(4)
lang_buck_mortality.columns = ['deaths', 'n', 'mortality_rate']
lang_buck_mortality['gap_pp'] = (lang_buck_mortality['mortality_rate'] - overall_mortality) * 100

print("\n" + lang_buck_mortality.to_string())
print(f"\nBucketed language gap: {lang_buck_mortality['gap_pp'].abs().max():.1f} pp")

# ============================================================================
# COMBINED: Race × Language
# ============================================================================
print("\n" + "=" * 80)
print("COMBINED DISPARITIES: Race × Language")
print("=" * 80)

combined_mortality = df.groupby(['race', 'language_bucketed']).agg({
    'mortality': ['sum', 'count', 'mean']
}).round(4)
combined_mortality.columns = ['deaths', 'n', 'mortality_rate']
combined_mortality['gap_pp'] = (combined_mortality['mortality_rate'] - overall_mortality) * 100
combined_mortality = combined_mortality.sort_values('gap_pp', ascending=False)

print("\nTop 10 race × language combinations by gap:")
print(combined_mortality.head(10).to_string())

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: What shows MORE disparity?")
print("=" * 80)

print(f"\n                                   | Max Gap | Mean Gap | Largest Single |")
print(f"-" * 76)
print(f"Race (5 groups)                    | {race_mortality['gap_pp'].abs().max():6.1f} pp | {race_mortality['gap_pp'].abs().mean():7.1f} pp | {race_mortality['gap_pp'].abs().max():6.1f} pp |")
print(f"Language bucketed (2 groups)       | {lang_buck_mortality['gap_pp'].abs().max():6.1f} pp | {lang_buck_mortality['gap_pp'].abs().mean():7.1f} pp | {lang_buck_mortality['gap_pp'].abs().max():6.1f} pp |")
print(f"Language raw ({df['language'].nunique()} groups) | {language_mortality['gap_pp'].abs().max():6.1f} pp | {language_mortality['gap_pp'].abs().mean():7.1f} pp | {language_mortality['gap_pp'].abs().max():6.1f} pp |")
print(f"Race × Language bucketed (10 combos)| {combined_mortality['gap_pp'].abs().max():6.1f} pp | {combined_mortality['gap_pp'].abs().mean():7.1f} pp | {combined_mortality['gap_pp'].abs().max():6.1f} pp |")

# ============================================================================
# STATISTICAL SIGNIFICANCE TESTS
# ============================================================================
print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

from scipy.stats import chi2_contingency

# Race vs mortality
race_mortality_ct = pd.crosstab(df['race'], df['mortality'])
chi2_race, p_race, dof_race, expected_race = chi2_contingency(race_mortality_ct)
print(f"\nRace vs Mortality: χ² = {chi2_race:.1f}, p = {p_race:.2e}")

# Language (bucketed) vs mortality
lang_mortality_ct = pd.crosstab(df['language_bucketed'], df['mortality'])
chi2_lang, p_lang, dof_lang, expected_lang = chi2_contingency(lang_mortality_ct)
print(f"Language vs Mortality: χ² = {chi2_lang:.1f}, p = {p_lang:.2e}")

# Race × Language vs mortality
race_lang_ct = pd.crosstab([df['race'], df['language_bucketed']], df['mortality'])
chi2_combined, p_combined, dof_combined, expected_combined = chi2_contingency(race_lang_ct)
print(f"Race × Language vs Mortality: χ² = {chi2_combined:.1f}, p = {p_combined:.2e}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if language_mortality['gap_pp'].abs().max() > race_mortality['gap_pp'].abs().max():
    print("\n✓ YES - Language shows LARGER disparities than race")
    print(f"  Language max gap: {language_mortality['gap_pp'].abs().max():.1f} pp")
    print(f"  Race max gap: {race_mortality['gap_pp'].abs().max():.1f} pp")
else:
    print("\n✗ NO - Race shows larger disparities than language")
    print(f"  Race max gap: {race_mortality['gap_pp'].abs().max():.1f} pp")
    print(f"  Language max gap: {language_mortality['gap_pp'].abs().max():.1f} pp")

print(f"\n  → This means language is a {p_lang/p_race:.1f}x stronger signal than race (by p-value)")
