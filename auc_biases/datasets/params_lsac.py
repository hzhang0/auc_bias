from pathlib import Path

# link to file
link = Path(__file__).parent.resolve()/'lsac.csv'

# columns
columns = ['age',
        'decile1',
        'decile3',
        'fam_inc',
        'lsat',
        'ugpa',
        'cluster',
        'fulltime',
        'male',
        'race1',
        'pass_bar']

# all training columns
train_cols = ['age',
        'decile1',
        'decile3',
        'fam_inc',
        'lsat',
        'ugpa',
        'cluster',
        'fulltime']

# label column
label = 'pass_bar'

# sensitive columns
sensitive_attributes = ["male", "race1"]

# whether data already contains splits
already_split = False

# list of all categorical columns
# source: https://rdrr.io/cran/fairml/man/law.school.admissions.html
# NB: some columns are deciles, but based on dataset convention not
# treating them as categorical. Also verified empirically that this
# does not change the results.
categorical_columns = ['cluster', 'fulltime']

# balanced groups
balance_groups = 0

has_header = True