import pm4py
from ml.core.model import AugmentationStrategyConfig
from ml.core.model import AugmentorConfig
from ml.augmentation.augmentation_strategy import AugmentationStrategyBuilder

EVENT_LOG_NAME = 'sepsis_cases_1.xes'

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
    augmentation_factor=1.5, # 1 = no augmentation, 2 = double the log size
    allow_multiple=True, # allow the augmentation of an already augmented trace
)

augmentation_strategy = AugmentationStrategyBuilder(config).build()

event_log = pm4py.read_xes(f'logs/{EVENT_LOG_NAME}', return_legacy_log_object=True)

augmented_event_log, augmentation_count, augmentation_record, augmentation_duration = augmentation_strategy.augment(
    event_log=event_log,
    record_augmentation=True,
    verbose=True,
)

event_log_filename, event_log_extension = EVENT_LOG_NAME.split('.')
augmented_filename = f'{event_log_filename}_AUGMENTED.{event_log_extension}'

pm4py.write_xes(event_log, f'output/{augmented_filename}')
