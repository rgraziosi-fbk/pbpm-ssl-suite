import os
import sys
import pm4py
import pandas as pd

from ml.core.model import AugmentationStrategyConfig
from ml.core.model import AugmentorConfig
from ml.augmentation.augmentation_strategy import AugmentationStrategyBuilder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

### Config ###
LOG_PATH = os.path.join('logs', 'bpic2012_2_EXTREME_cts_TRAIN.csv')

AUGMENTATION_FACTOR = 10

NUM_GENERATIONS = 10
GENERATION_CONFIG = {
  'deviant': 203,
  'regular': 0,
}


def generate_log_kappel(
  generation_config,
  augmentation_strategy,
  log_path,
  csv_sep=';',
  trace_key='case:concept:name',
  activity_key='concept:name',
  timestamp_key='time:timestamp',
  label_key='label',
  output_path='output',
  output_name='gen_kappel',
):
  if output_path and not os.path.exists(output_path):
    os.makedirs(output_path)

  generated_dataframes = []

  for i, (label, num_to_generate) in enumerate(generation_config.items()):
    if num_to_generate == 0: continue

    dataframe = pd.read_csv(log_path, sep=csv_sep)
    dataframe = pm4py.format_dataframe(dataframe, case_id=trace_key, activity_key=activity_key, timestamp_key=timestamp_key)

    # filter dataframe by label
    dataframe_label = dataframe[dataframe[label_key] == label]

    # convert DataFrame to EventLog (because augmentation lib requires this type of object)
    log_label = pm4py.convert_to_event_log(dataframe_label)

    # Augment label cases
    log_label_augmented, _, _, _ = augmentation_strategy.augment(
      event_log=log_label,
      record_augmentation=True,
      verbose=True,
    )
    dataframe_label_augmented = pm4py.convert_to_dataframe(log_label_augmented)
    dataframe_label_only_augmented = dataframe_label_augmented[dataframe_label_augmented['case:creator'].isin(AUGMENTORS)]
    label_cases = dataframe_label_only_augmented['case:concept:name'].unique().tolist()[:num_to_generate]
    dataframe_label_only_augmented = dataframe_label_only_augmented[dataframe_label_only_augmented['case:concept:name'].isin(label_cases)]

    # append i to case id to avoid duplicate case ids
    dataframe_label_only_augmented['case:concept:name'] = dataframe_label_only_augmented['case:concept:name'] + f'_{i}'

    generated_dataframes.append(dataframe_label_only_augmented)

  generated_dataframe = pd.concat(generated_dataframes)

  generated_dataframe_cases = generated_dataframe['case:concept:name'].unique().tolist()
  assert len(generated_dataframe_cases) == generation_config['deviant'] + generation_config['regular']

  # save log as both xes and csv
  generated_dataset_path = os.path.join(output_path, f'{output_name}.xes')
  generated_dataframe = generated_dataframe.drop(columns=[trace_key])
  generated_dataframe['time:timestamp'] = pd.to_datetime(generated_dataframe['time:timestamp'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize(None)
  pm4py.write_xes(generated_dataframe, generated_dataset_path, case_id_key='case:concept:name')
  generated_dataframe.to_csv(generated_dataset_path.replace('.xes', '.csv'), sep=csv_sep)

  print(f'Generated log saved at {generated_dataset_path}')
  return generated_dataset_path


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
    augmentation_factor=AUGMENTATION_FACTOR, # 1 = no augmentation, 2 = double the log size
    allow_multiple=True, # allow the augmentation of an already augmented trace
)

augmentation_strategy = AugmentationStrategyBuilder(config).build()


for i in range(1, NUM_GENERATIONS+1):
  generate_log_kappel(
    GENERATION_CONFIG,
    augmentation_strategy=augmentation_strategy,
    log_path=LOG_PATH,
    csv_sep=';',
    trace_key='Case ID',
    activity_key='Activity',
    timestamp_key='time:timestamp',
    label_key='label',
    output_path=os.path.join('output'),
    output_name=f'gen_kappel_{i}'
  )
