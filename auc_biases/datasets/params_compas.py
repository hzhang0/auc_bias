# link to file
link = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"

# columns
columns = ['sex','age','race','juv_fel_count','juv_misd_count','juv_other_count','priors_count','c_charge_degree','low_risk']

# all training columns
train_cols = ['age','juv_fel_count','juv_misd_count','juv_other_count','priors_count','c_charge_degree']

# label column
label = 'is_recid'

# sensitive columns
sensitive_attributes = ["sex", "race"]

# whether data already contains splits
already_split = False

# list of all categorical columns
categorical_columns = ['c_charge_degree']

has_header = True
