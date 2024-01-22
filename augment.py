import pm4py
import pandas as pd
from ml.core.model import AugmentationStrategyConfig
from ml.core.model import AugmentorConfig
from ml.augmentation.augmentation_strategy import AugmentationStrategyBuilder

EVENT_LOG_NAME = 'sepsis_cases_1.csv'
LABEL_TO_AUGMENT = 'deviant'
CASE_ID_KEY = 'Case ID'
ACTIVITY_KEY = 'Activity'
TIMESTAMP_KEY = 'time:timestamp'
LABEL_KEY = 'label'

# Read log in pandas DataFrame
dataframe = pd.read_csv(f'logs/{EVENT_LOG_NAME}', sep=';')

# Format pandas DataFrame (this will add a column named case:concept:name copying values from column CASE_ID_KEY)
dataframe = pm4py.format_dataframe(dataframe, case_id=CASE_ID_KEY, activity_key=ACTIVITY_KEY, timestamp_key=TIMESTAMP_KEY)

# Filter DataFrame by label
dataframe_label, dataframe_no_label = dataframe[dataframe[LABEL_KEY] == LABEL_TO_AUGMENT], dataframe[dataframe[LABEL_KEY] != LABEL_TO_AUGMENT]

# Convert DataFrame to EventLog (because augmentation lib requires this type of object)
log_label = pm4py.convert_to_event_log(dataframe_label)

# Perform augmentation
AUGMENTORS = [
    'RandomInsertion',
    'RandomDeletion',
    'ParallelSwap',
    'FragmentAugmentation',
    'ReworkActivity',
    'DeleteReworkActivity',
    'RandomReplacement',
    'RandomSwap',
    # 'LoopAugmentation',
]

augmentors = []
for augmentor_name in AUGMENTORS:
    parameters = {}

    if augmentor_name == 'LoopAugmentation':
        parameters = {
            'max_additional_repetitions': 3,
            'duration_tolerance': 0.2, # duration of each additional repetition can be anywhere between 80% and 120% of original duration
        }

    augmentor_config = AugmentorConfig(name=augmentor_name, parameters=parameters)
    augmentors.append(augmentor_config)

config = AugmentationStrategyConfig(
    id=1,
    name='mixed', # 'single' = apply just the first provided augmentor, 'mixed' = randomly apply all augmentors provided
    seed=42,
    augmentors=augmentors,
    augmentation_factor=2, # 1 = no augmentation, 2 = double the log size
    allow_multiple=True, # allow the augmentation of an already augmented trace
)

augmentation_strategy = AugmentationStrategyBuilder(config).build()

log_augmented, augmentation_count, augmentation_record, augmentation_duration = augmentation_strategy.augment(
    event_log=log_label,
    record_augmentation=True,
    verbose=True,
)
dataframe_augmented = pm4py.convert_to_dataframe(log_augmented)

# log_augmented = log_label + traces generated with augmentation

# Re-add traces that were filtered out to augmented log (those with label != LABEL_TO_AUGMENT)
dataframe = pd.concat([dataframe_augmented, dataframe_no_label], ignore_index=True)
dataframe[CASE_ID_KEY] = dataframe['case:concept:name']


log_filename, log_extension = EVENT_LOG_NAME.split('.')
augmented_filename = f'{log_filename}_AUGMENTED.{log_extension}'

dataframe.to_csv(f'output/{augmented_filename}', sep=';', index=False)
