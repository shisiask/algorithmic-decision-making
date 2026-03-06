#!/usr/bin/env python3

"""
File: q3_comparison.py
----------------------
This script trains two Perceptron models:
  1. Without protected features (age, gender, race)
  2. With protected features included
Then prints a demographic comparison table matching the Q3 format.
"""

import constants
from dataset import Dataset
from perceptronmodel import PerceptronModel

# ── File names ──────────────────────────────────────────────────────────────
# Update these if your CSV filenames are different
TRAINING_DATAFILE = "recidivism-training-data.csv"
TESTING_DATAFILE = "recidivism-testing-data.csv"


# ── Feature selection helpers ───────────────────────────────────────────────

def get_base_features():
    """Returns the list of non-protected feature indexes (same as the original code)."""
    features = []

    # Juvenile felony count
    features.append(constants.JUVENILE_FELONY_COUNT_0)
    features.append(constants.JUVENILE_FELONY_COUNT_1)
    features.append(constants.JUVENILE_FELONY_COUNT_2)
    features.append(constants.JUVENILE_FELONY_COUNT_3_OR_MORE)

    # Juvenile misdemeanor count
    features.append(constants.JUVENILE_MISDEMEANOR_COUNT_0)
    features.append(constants.JUVENILE_MISDEMEANOR_COUNT_1)
    features.append(constants.JUVENILE_MISDEMEANOR_COUNT_2)
    features.append(constants.JUVENILE_MISDEMEANOR_COUNT_3_OR_MORE)

    # Juvenile other offense count
    features.append(constants.JUVENILE_OTHER_COUNT_0)
    features.append(constants.JUVENILE_OTHER_COUNT_1)
    features.append(constants.JUVENILE_OTHER_COUNT_2)
    features.append(constants.JUVENILE_OTHER_COUNT_3_OR_MORE)

    # Prior conviction count
    features.append(constants.PRIOR_CONVICTIONS_COUNT_0)
    features.append(constants.PRIOR_CONVICTIONS_COUNT_1)
    features.append(constants.PRIOR_CONVICTIONS_COUNT_2)
    features.append(constants.PRIOR_CONVICTIONS_COUNT_3_OR_MORE)

    # Charge degree
    features.append(constants.CHARGE_DEGREE_FELONY)
    features.append(constants.CHARGE_DEGREE_MISDEMEANOR)

    # Charge description
    features.append(constants.CHARGE_DESC_NO_CHARGE)
    features.append(constants.CHARGE_DESC_LICENSE_ISSUE)
    features.append(constants.CHARGE_DESC_PUBLIC_DISTURBANCE)
    features.append(constants.CHARGE_DESC_NEGLIGENCE)
    features.append(constants.CHARGE_DESC_DRUG_RELATED)
    features.append(constants.CHARGE_DESC_ALCOHOL_RELATED)
    features.append(constants.CHARGE_DESC_WEAPONS_RELATED)
    features.append(constants.CHARGE_DESC_EVADING_ARREST)
    features.append(constants.CHARGE_DESC_NONVIOLENT_HARM)
    features.append(constants.CHARGE_DESC_THEFT_FRAUD_BURGLARY)
    features.append(constants.CHARGE_DESC_LEWDNESS_PROSTITUTION)
    features.append(constants.CHARGE_DESC_VIOLENT_CRIME)

    return features


def get_all_features():
    """Returns base features + all protected features (age, gender, race)."""
    features = get_base_features()

    # Age
    features.append(constants.AGE_LESS_THAN_25)
    features.append(constants.AGE_25_TO_45)
    features.append(constants.AGE_GREATER_THAN_45)

    # Gender
    features.append(constants.GENDER_FEMALE)
    features.append(constants.GENDER_MALE)

    # Race
    features.append(constants.RACE_OTHER)
    features.append(constants.RACE_ASIAN)
    features.append(constants.RACE_NATIVE_AMERICAN)
    features.append(constants.RACE_CAUCASIAN)
    features.append(constants.RACE_HISPANIC)
    features.append(constants.RACE_AFRICAN_AMERICAN)

    return features


# ── Statistics computation ──────────────────────────────────────────────────

def compute_stats(perceptronmodel, data, feature_index=None):
    """
    Computes confusion matrix counts for the given model on the given data.
    If feature_index is provided, only considers instances where that feature == 1.
    Returns a dictionary with TP, TN, FP, FN counts and derived percentages.
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(data.get_size()):
        instance = data.get_instance(i)

        # Filter to subgroup if feature_index is specified
        if (feature_index is not None) and (instance[feature_index] != 1):
            continue

        prediction = perceptronmodel.predict(instance)
        actual_output = data.get_output(i)

        if prediction == 1:
            if actual_output == 1:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if actual_output == 0:
                true_negative += 1
            else:
                false_negative += 1

    total = true_positive + true_negative + false_positive + false_negative

    # Avoid division by zero
    if total == 0:
        return None

    stats = {
        "TP": true_positive,
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
        "Total": total,
        "TP%": true_positive / total,
        "FP%": false_positive / total,
        "TN%": true_negative / total,
        "FN%": false_negative / total,
        "PPR": (true_positive + false_positive) / total,          # Detention rate
        "Precision": true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0,
        "Accuracy": (true_positive + true_negative) / total,
    }
    return stats


# ── Printing helpers ────────────────────────────────────────────────────────

def print_full_results(perceptronmodel, data, label=""):
    """Prints the same output as the original print_results for the full dataset."""
    stats = compute_stats(perceptronmodel, data)
    if stats is None:
        print("  No data.")
        return
    print(label)
    print(f"  Overall accuracy = {stats['Accuracy']:.4f}")
    print(f"  True Positives:  {stats['TP']}")
    print(f"  True Negatives:  {stats['TN']}")
    print(f"  False Positives: {stats['FP']}")
    print(f"  False Negatives: {stats['FN']}")
    print(f"  Of detained, % who would NOT have recidivated = {stats['FP'] / (stats['TP'] + stats['FP']):.4f}")
    print(f"  Of released, % who recidivated              = {stats['FN'] / (stats['TN'] + stats['FN']):.4f}")
    print(f"  Of non-recidivists, % detained              = {stats['FP'] / (stats['TN'] + stats['FP']):.4f}")
    print(f"  Of recidivists, % released                  = {stats['FN'] / (stats['TP'] + stats['FN']):.4f}")
    print()


def print_comparison_table(model_without, model_with, data):
    """
    Prints a formatted comparison table of both models across demographic subgroups.
    """
    # Define the subgroups to compare
    groups = [
        ("Caucasian",       constants.RACE_CAUCASIAN),
        ("African American", constants.RACE_AFRICAN_AMERICAN),
        ("Age < 25",        constants.AGE_LESS_THAN_25),
        ("Age 25-45",       constants.AGE_25_TO_45),
        ("Age > 45",        constants.AGE_GREATER_THAN_45),
        ("Female",          constants.GENDER_FEMALE),
        ("Male",            constants.GENDER_MALE),
    ]

    # Column headers
    cols = ["TP%", "FP%", "TN%", "FN%", "PPR", "Prec", "Acc"]

    # Print header
    print("=" * 130)
    print(f"{'Q3 COMPARISON TABLE — TESTING DATA':^130}")
    print("=" * 130)
    print()
    print(f"{'':18s}|{'MODEL WITHOUT PROTECTED FEATURES':^55}|{'MODEL WITH PROTECTED FEATURES':^55}")
    print(f"{'Group':18s}|", end="")
    for c in cols:
        print(f" {c:>6s}", end="")
    print("  |", end="")
    for c in cols:
        print(f" {c:>6s}", end="")
    print()
    print("-" * 130)

    # Print each group row
    for group_name, feature_index in groups:
        stats_wo = compute_stats(model_without, data, feature_index)
        stats_w = compute_stats(model_with, data, feature_index)

        if stats_wo is None or stats_w is None:
            print(f"{group_name:18s}|  Insufficient data")
            continue

        print(f"{group_name:18s}|", end="")
        print(f" {stats_wo['TP%']:6.2%}", end="")
        print(f" {stats_wo['FP%']:6.2%}", end="")
        print(f" {stats_wo['TN%']:6.2%}", end="")
        print(f" {stats_wo['FN%']:6.2%}", end="")
        print(f" {stats_wo['PPR']:6.2%}", end="")
        print(f" {stats_wo['Precision']:6.2%}", end="")
        print(f" {stats_wo['Accuracy']:6.2%}", end="")
        print("  |", end="")
        print(f" {stats_w['TP%']:6.2%}", end="")
        print(f" {stats_w['FP%']:6.2%}", end="")
        print(f" {stats_w['TN%']:6.2%}", end="")
        print(f" {stats_w['FN%']:6.2%}", end="")
        print(f" {stats_w['PPR']:6.2%}", end="")
        print(f" {stats_w['Precision']:6.2%}", end="")
        print(f" {stats_w['Accuracy']:6.2%}", end="")
        print()

    print("=" * 130)
    print()
    print("Legend:")
    print("  TP%  = True Positives  / Group Total")
    print("  FP%  = False Positives / Group Total")
    print("  TN%  = True Negatives  / Group Total")
    print("  FN%  = False Negatives / Group Total")
    print("  PPR  = Positive Prediction Rate (Detention Rate) = (TP + FP) / Total")
    print("  Prec = Precision (PPV) = TP / (TP + FP)")
    print("  Acc  = Accuracy = (TP + TN) / Total")


# ── Raw counts table ────────────────────────────────────────────────────────

def print_raw_counts(model, data, label=""):
    """Prints raw confusion matrix counts per subgroup for one model."""
    groups = [
        ("Overall",          None),
        ("Caucasian",        constants.RACE_CAUCASIAN),
        ("African American", constants.RACE_AFRICAN_AMERICAN),
        ("Age < 25",         constants.AGE_LESS_THAN_25),
        ("Age 25-45",        constants.AGE_25_TO_45),
        ("Age > 45",         constants.AGE_GREATER_THAN_45),
        ("Female",           constants.GENDER_FEMALE),
        ("Male",             constants.GENDER_MALE),
    ]

    print(f"\n{label}")
    print(f"{'Group':18s} | {'TP':>6s} {'TN':>6s} {'FP':>6s} {'FN':>6s} {'Total':>6s}")
    print("-" * 60)
    for group_name, feature_index in groups:
        stats = compute_stats(model, data, feature_index)
        if stats is None:
            print(f"{group_name:18s} |  No data")
            continue
        print(f"{group_name:18s} | {stats['TP']:6d} {stats['TN']:6d} {stats['FP']:6d} {stats['FN']:6d} {stats['Total']:6d}")
    print()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Load data
    print("Reading training data...")
    training_set = Dataset(TRAINING_DATAFILE)
    print("Reading testing data...")
    testing_set = Dataset(TESTING_DATAFILE)

    # ── Model 1: Without protected features ─────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING MODEL WITHOUT PROTECTED FEATURES")
    print("=" * 60)
    features_base = get_base_features()
    model_without = PerceptronModel(training_set, features_base)

    print_full_results(model_without, training_set, "\nTraining Set Results (Without Protected):")
    print_full_results(model_without, testing_set, "Testing Set Results (Without Protected):")
    print_raw_counts(model_without, testing_set, "Raw Counts — Without Protected (Testing Data):")

    # ── Model 2: With protected features ────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING MODEL WITH PROTECTED FEATURES")
    print("=" * 60)
    features_all = get_all_features()
    model_with = PerceptronModel(training_set, features_all)

    print_full_results(model_with, training_set, "\nTraining Set Results (With Protected):")
    print_full_results(model_with, testing_set, "Testing Set Results (With Protected):")
    print_raw_counts(model_with, testing_set, "Raw Counts — With Protected (Testing Data):")

    # ── Side-by-side comparison table ───────────────────────────────────────
    print()
    print_comparison_table(model_without, model_with, testing_set)


if __name__ == '__main__':
    main()