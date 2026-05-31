from scipy.io import loadmat

mat = loadmat("../data/raw/DREAMER.mat")

print("\nTOP LEVEL KEYS")
for key in mat:
    if not key.startswith("__"):
        print(key)

dreamer = mat["DREAMER"]

print("\nDREAMER STRUCT")
print("Type:", type(dreamer))
print("Shape:", dreamer.shape)
print("Fields:", dreamer.dtype.names)

print("\nDATASET METADATA")
print("Subjects:", dreamer["noOfSubjects"][0, 0])
print("Video Sequences:", dreamer["noOfVideoSequences"][0, 0])
print("EEG Sampling Rate:", dreamer["EEG_SamplingRate"][0, 0])
print("ECG Sampling Rate:", dreamer["ECG_SamplingRate"][0, 0])

print("\nEEG ELECTRODES")
electrodes = dreamer["EEG_Electrodes"][0, 0]
for idx, electrode in enumerate(electrodes[0], start=1):
    print(f"{idx:02d}: {electrode[0]}")

data = dreamer["Data"][0, 0]

print("\nDATA FIELD")
print("Type:", type(data))
print("Shape:", data.shape)
print("Dtype:", data.dtype)

subject1 = data[0, 0]

print("\nSUBJECT STRUCTURE")
print("Fields:", subject1.dtype.names)

print("\nSUBJECT METADATA")
print("Age:", subject1["Age"])
print("Gender:", subject1["Gender"])

eeg = subject1["EEG"][0, 0]

print("\nEEG STRUCTURE")
print("Fields:", eeg.dtype.names)

print("\nBASELINE EEG")
print("Trials Shape:", eeg["baseline"][0, 0].shape)
print(
    "Trial 1 Shape:",
    eeg["baseline"][0, 0][0, 0].shape
)

print("\nSTIMULUS EEG")
print("Trials Shape:", eeg["stimuli"][0, 0].shape)
print(
    "Trial 1 Shape:",
    eeg["stimuli"][0, 0][0, 0].shape
)

print("\nLABEL STRUCTURES")
print(
    "Valence Shape:",
    subject1["ScoreValence"][0, 0].shape
)
print(
    "Arousal Shape:",
    subject1["ScoreArousal"][0, 0].shape
)
print(
    "Dominance Shape:",
    subject1["ScoreDominance"][0, 0].shape
)

print("\nVALENCE SCORES")
print(subject1["ScoreValence"][0, 0].flatten())

print("\nAROUSAL SCORES")
print(subject1["ScoreArousal"][0, 0].flatten())

print("\nDOMINANCE SCORES")
print(subject1["ScoreDominance"][0, 0].flatten())

print("\nDATASET SUMMARY")
print(f"Subjects: {data.shape[1]}")
print(
    f"Trials per Subject: "
    f"{subject1['ScoreValence'][0,0].shape[0]}"
)
print(
    f"Total Samples: "
    f"{data.shape[1] * subject1['ScoreValence'][0,0].shape[0]}"
)
print(
    f"EEG Channels: "
    f"{eeg['stimuli'][0,0][0,0].shape[1]}"
)