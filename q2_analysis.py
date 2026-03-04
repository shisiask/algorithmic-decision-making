import constants
from dataset import Dataset
from perceptronmodel import PerceptronModel

TRAINING_DATAFILE = "recidivism-training-data.csv"
TESTING_DATAFILE = "recidivism-testing-data.csv"

def get_stats(perceptronmodel, data, feature_index=None):
    tp = tn = fp = fn = 0
    for i in range(data.get_size()):
        instance = data.get_instance(i)
        if (feature_index is None) or (instance[feature_index] == 1):
            prediction = perceptronmodel.predict(instance)
            actual = data.get_output(i)
            if prediction == 1:
                if actual == 1: tp += 1
                else: fp += 1
            else:
                if actual == 0: tn += 1
                else: fn += 1
    return tp, tn, fp, fn

def print_fairness_stats(label, tp, tn, fp, fn):
    total = tp + tn + fp + fn
    if total == 0:
        print(f"{label}: No data")
        return
    accuracy = (tp + tn) / total
    # FPR = false positive rate among non-recidivists (wrongly detained)
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    # FNR = false negative rate among recidivists (wrongly released)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    # PPV = precision (of those detained, how many actually recidivate)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    # NPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    # Positive prediction rate (disparate impact)
    ppr = (tp + fp) / total

    print(f"\n--- {label} ---")
    print(f"  Total: {total}  |  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"  Accuracy:                          {accuracy:.4f}")
    print(f"  False Positive Rate (FPR):         {fpr:.4f}  [of non-recidivists, % wrongly detained]")
    print(f"  False Negative Rate (FNR):         {fnr:.4f}  [of recidivists, % wrongly released]")
    print(f"  Precision (PPV):                   {ppv:.4f}  [of detained, % who would recidivate]")
    print(f"  NPV:                               {npv:.4f}  [of released, % who would not recidivate]")
    print(f"  Positive Prediction Rate:          {ppr:.4f}  [% of group detained]")

# Train model (no protected features)
training_set = Dataset(TRAINING_DATAFILE)
testing_set = Dataset(TESTING_DATAFILE)

features_to_use = list(range(30))  # indices 0-29 = non-protected features

pm = PerceptronModel(training_set, features_to_use)

print("=" * 60)
print("TRAINING DATA")
print("=" * 60)
tp, tn, fp, fn = get_stats(pm, training_set)
print_fairness_stats("Overall (Training)", tp, tn, fp, fn)

tp, tn, fp, fn = get_stats(pm, training_set, constants.RACE_AFRICAN_AMERICAN)
print_fairness_stats("African American (Training)", tp, tn, fp, fn)

tp, tn, fp, fn = get_stats(pm, training_set, constants.RACE_CAUCASIAN)
print_fairness_stats("Caucasian (Training)", tp, tn, fp, fn)

print("\n")
print("=" * 60)
print("TESTING DATA")
print("=" * 60)
tp, tn, fp, fn = get_stats(pm, testing_set)
print_fairness_stats("Overall (Testing)", tp, tn, fp, fn)

tp, tn, fp, fn = get_stats(pm, testing_set, constants.RACE_AFRICAN_AMERICAN)
print_fairness_stats("African American (Testing)", tp, tn, fp, fn)

tp, tn, fp, fn = get_stats(pm, testing_set, constants.RACE_CAUCASIAN)
print_fairness_stats("Caucasian (Testing)", tp, tn, fp, fn)

