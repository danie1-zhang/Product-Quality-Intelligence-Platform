

SUPPORTED_LABELS = [
     "NO_COMPLAINT",
    "FUNCTIONALITY",
    "BUILD_QUALITY",
    "SHIPPING",
    "FIT_COMPATIBILITY",
    "USABILITY_SETUP",
]

LABEL_TO_ID, ID_TO_LABEL = {}, {}
for idx, label in enumerate(SUPPORTED_LABELS):
    LABEL_TO_ID[label] = idx
    ID_TO_LABEL[idx] = label