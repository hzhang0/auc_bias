# link to file
link = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# columns
columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]

# all training columns
train_cols = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", 
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry"]

# label column
label = 'Income'

# sensitive columns
sensitive_attributes = ["Gender","Race"]

# whether data already contains splits
already_split = False

# list of all categorical columns
categorical_columns = ["WorkClass", "Education",
                       "MaritalStatus", "Occupation", "Relationship",
                       "NativeCountry"]

has_header = False