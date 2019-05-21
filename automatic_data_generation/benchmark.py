from snips_nlu import SnipsNLUEngine
from snips_nlu_metrics import compute_train_test_metrics

datadir = os.path.join(*args.train_path.split('/')[:-1])
csv2json(datadir, datadir, augmented=False)
csv2json(datadir, datadir, augmented=True)

print('Starting benchmarking...')

def my_matching_lambda(lhs_slot, rhs_slot):
    return lhs_slot['text'].strip() == rhs_slot["rawValue"].strip()

raw_metrics = compute_train_test_metrics(train_dataset="data/train.json",
                                        test_dataset="data/validate.json",
                                        engine_class=SnipsNLUEngine,
                                        slot_matching_lambda = my_matching_lambda
                                        )
augmented_metrics = compute_train_test_metrics(train_dataset="data/train_augmented.json",
                                        test_dataset="data/validate.json",
                                        engine_class=SnipsNLUEngine,
                                        slot_matching_lambda = my_matching_lambda
                                        )

print('----------METRICS----------')
print('Without augmentation : ')
print(raw_metrics['average_metrics'])
print('With augmentation : ')
print(augmented_metrics['average_metrics'])
intent_improvement = 100 * ((augmented_metrics['average_metrics']['intent']['f1'] - raw_metrics['average_metrics']['intent']['f1'])
                            / raw_metrics['average_metrics']['intent']['f1'])
slot_improvement = 100 * ((augmented_metrics['average_metrics']['slot']['f1'] - raw_metrics['average_metrics']['slot']['f1'])
                            / raw_metrics['average_metrics']['slot']['f1'])
score = intent_improvement + slot_improvement

print('Improvement metrics : intent {:.4f} slot {:.4f} total {:.4f}'.format(intent_improvement, slot_improvement, score))

run['metrics'] = {'raw':raw_metrics, 'augmented':augmented_metrics}
