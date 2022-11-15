from ml.core import model
from ml.persistence import json

if __name__ == '__main__':
    datasets = [
        model.Dataset('Helpdesk', r'/home/ai4-admin/event_logs/Helpdesk.xes')
    ]

    approaches = [
        model.Approach('Buksh', 'process-transformer', '/home/ai4-admin/workspace/pbpm-processtransformer/main.py', {
            'task': 'next_activity',
            'epochs': 1,
            'batch_size': 12,
            'learning_rate': 0.001,
            'gpu': 0,
        }),
        model.Approach('Pasquadibisceglie', 'pasquadibisceglie', '/home/ai4-admin/workspace/pbpm-pasquadibisceglie/main.py', {
            'batch_size': 128,
            'epochs': 1,
            'n_layers': 2,
            'reg': 0.0001,
            'validation_split': 0.2,
            'patience': 6
        }),
        model.Approach('Camargo', 'camargo', '/home/ai4-admin/workspace/pbpm-camargo/main.py', {
            'n_gram_size': 5,
            'model_type': 'shared_categorical',
            'l_size': 100,
            'imp': 2,
            'lstm_act': 'tanh',
            'optim': 'Nadam',
            'epochs': 1,
            'emb_epochs': 1,
            'batch_size': 32,
            'dense_act': None,
            'rp_sim': 0.85
        }),
        model.Approach('Theis', 'theis-pydream', '/home/ai4-admin/workspace/pbpm-theis/main.py', {
            'seed': 1,
            'epochs': 1,
            'batch_size': 64,
            'dropout_rate': 0.2,
            'eval_size': 0.1,
            'activation_function': 'relu'
        }),
        model.Approach('Tax', 'tax', '/home/ai4-admin/workspace/pbpm-tax/main.py', {
            'prediction_task': "ACT_TIME",
            'epochs': 1,
            'validation_split': 0.2,
            'learning_rate': 0.002,
            'schedule_decay': 0.004,
            'clipvalue': 3,
            'patience': 10,
            'prediction_size': 3
        }),
        model.Approach('Khan', 'khan', '/home/ai4-admin/workspace/pbpm-khan/main.py', {
            'iterations': 1,
            'hidden_controller_dim': 100,
            'use_emb': False,
            'use_mem': True,
            'decoder_mode': True,
            'dual_controller': True,
            'write_protect': True,
            'attend_dim': 0,
            'batch_size': 1,
            'sequence_max_length': 100,
            'words_count': 5,
            'word_size': 20,
            'read_heads': 1
        }),
        model.Approach('Mauro', 'mauro', '/home/ai4-admin/workspace/pbpm-mauro/main.py', {
            'iterations': 1,
            'model_type': 'ACT',
            'epochs': 1,
            'validation_split': 0.2,
            'patience': 20
        }),
    ]

    splitter_configuration = model.SplitterConfiguration(name='time',
                                                         training_size=0.7,
                                                         by='first',
                                                         seeds=[42, 567, 789],
                                                         repetitions=3,
                                                         folds=2)

    augmentor_names = [model.AugmentorConfig('RandomInsertion', {}),
                       model.AugmentorConfig('RandomDeletion', {}),
                       model.AugmentorConfig('ParallelSwap', {}),
                       model.AugmentorConfig('FragmentAugmentation', {}),
                       model.AugmentorConfig('ReworkActivity', {}),
                       model.AugmentorConfig('DeleteReworkActivity', {}),
                       model.AugmentorConfig('RandomReplacement', {}),
                       model.AugmentorConfig('RandomSwap', {}),
                       model.AugmentorConfig('LoopAugmentation', {'max_additional_repetitions': 3,
                                                                  'duration_tolerance': 0.2})
                       ]

    augmentation_strategies_config = [
        model.AugmentationStrategyConfig(1, 'mixed', 42, augmentor_names, 1.2, True),
    ]

    experiment = model.AugmentationExperiment('aug_study_test', 'data', '/home/ai4-admin/runs',
                                              '/home/ai4-admin/ev', datasets, approaches,
                                              splitter_configuration, 2, augmentation_strategies_config)

    exporter = json.JsonExperimentExporter(r'/home/ai4-admin/runs/test_exp_small.json')
    exporter.save(experiment)

    importer = json.JsonExperimentImporter(r'/home/ai4-admin/runs/test_exp_small.json')
    loaded_experiment = importer.load()





