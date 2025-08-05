# link to file
from pathlib import Path

# link to file
link = Path(__file__).parent.resolve()/'mimic_mortality_tabular.csv'

# all training columns
train_cols = ['1_Diastolic blood pressure mean', '2_Diastolic blood pressure mean',
       '3_Diastolic blood pressure mean', '4_Diastolic blood pressure mean',
       '1_Fraction inspired oxygen mean', '2_Fraction inspired oxygen mean',
       '3_Fraction inspired oxygen mean', '4_Fraction inspired oxygen mean',
       '1_Glascow coma scale total mean', '2_Glascow coma scale total mean',
       '3_Glascow coma scale total mean', '4_Glascow coma scale total mean',
       '1_Glucose mean', '2_Glucose mean', '3_Glucose mean', '4_Glucose mean',
       '1_Heart Rate mean', '2_Heart Rate mean', '3_Heart Rate mean',
       '4_Heart Rate mean', '1_Mean blood pressure mean',
       '2_Mean blood pressure mean', '3_Mean blood pressure mean',
       '4_Mean blood pressure mean', '1_Oxygen saturation mean',
       '2_Oxygen saturation mean', '3_Oxygen saturation mean',
       '4_Oxygen saturation mean', '1_Respiratory rate mean',
       '2_Respiratory rate mean', '3_Respiratory rate mean',
       '4_Respiratory rate mean', '1_Systolic blood pressure mean',
       '2_Systolic blood pressure mean', '3_Systolic blood pressure mean',
       '4_Systolic blood pressure mean', '1_Temperature mean',
       '2_Temperature mean', '3_Temperature mean', '4_Temperature mean',
       '1_Weight mean', '2_Weight mean', '3_Weight mean', '4_Weight mean',
       '1_pH mean', '2_pH mean', '3_pH mean', '4_pH mean', 'Age']

# label column
label = 'target'

# sensitive columns
sensitive_attributes = ['Gender', 'Race']

columns = train_cols + [label] + sensitive_attributes + ['fold_id', 'ID']

# whether data already contains splits
already_split = True

# list of all categorical columns
categorical_columns = []

# balanced groups
balance_groups = 0

has_header = True

