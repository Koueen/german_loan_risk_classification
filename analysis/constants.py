COLUMN_NAMES = [
    'acc_status',
    'duration_months',
    'credit_history',
    'purpose',
    'credit_amount',
    'savings',
    'present_employment',
    'installment_rate',
    'personal_sex_status',
    'guarantors',
    'present_residence',
    'properties',
    'age',
    'other_installment_plans',
    'housing',
    'n_credits',
    'job_status',
    'liable_support',
    'telephone',
    'foreign',
    'risk',
]


NUMERICAL_COLUMNS = [
    'duration_months',
    'credit_amount',
    'installment_rate',
    'present_residence',
    'age',
    'n_credits',
    'liable_support',
]
CATEGORICAL_COLUMNS = [
    'acc_status',
    'credit_history',
    'purpose',
    'savings',
    'present_employment',
    'personal_sex_status',
    'guarantors',
    'properties',
    'other_installment_plans',
    'housing',
    'job_status',
    'telephone',
    'foreign',
]

RISK_COLOR_MAP = {'good': 'rgb(196,166,44)', 'bad': 'rgb(56,41,131)'}

SEED = 2
