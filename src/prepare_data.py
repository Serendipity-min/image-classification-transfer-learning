"""P0 数据完整性与固定分层划分；允许检查 test 结构，禁止计算模型测试成绩。"""

from collections import defaultdict
import hashlib
import json
import tarfile

import numpy as np
from torchvision.datasets import CIFAR100, SVHN

from tlcls.common import ROOT, atomic_json, load_json, provenance, sha256
from tlcls.data import official_train, raw_arrays


def main():
    assets = load_json(ROOT / 'records/assets.json')
    assert assets['status'] == 'completed', '数据或权重尚未完整就绪'
    for entry in assets['assets']:
        path = ROOT / entry['remote']
        assert path.stat().st_size == entry['size_bytes'] and sha256(path) == entry['sha256']
    # 只解压通过官方 MD5 的档案；data filter 拒绝路径逃逸和设备文件。
    with tarfile.open(ROOT / 'data/cifar-100-python.tar.gz') as archive:
        archive.extractall(ROOT / 'data', filter='data')
    receipt = {'provenance': provenance(), 'datasets': {}}
    for dataset, classes, ntrain, nval, ntest in [
            ('cifar100', 100, 50000, 5000, 10000), ('svhn', 10, 73257, 7326, 26032)]:
        raw = official_train(dataset)
        images, labels = raw_arrays(raw, dataset)
        assert len(images) == ntrain and set(labels.tolist()) == set(range(classes))
        counts = np.bincount(labels, minlength=classes)
        if dataset == 'cifar100':
            quotas = np.full(classes, 50)
        else:
            # 使用整数余数消除浮点并列歧义，严格补足 7326 个验证样本。
            numerators = nval * counts
            quotas = numerators // ntrain
            remaining = nval - int(quotas.sum())
            ranking = sorted(range(classes), key=lambda c: (-int(numerators[c] % ntrain), c))
            quotas[ranking[:remaining]] += 1
        rng = np.random.default_rng(20261020)
        validation, training = [], []
        for label in range(classes):
            indices = rng.permutation(np.flatnonzero(labels == label))
            validation.extend(indices[:quotas[label]].tolist())
            training.extend(indices[quotas[label]:].tolist())
        training, validation = sorted(training), sorted(validation)
        assert len(validation) == nval and len(training) == ntrain - nval
        assert not (set(training) & set(validation))
        assert set(training) | set(validation) == set(range(ntrain))
        display = []
        for label in range(classes):
            display += [f'{dataset}/train/{index}' for index in validation if labels[index] == label][:2]
            if len(display) >= 20:
                break
        manifest = {'dataset': dataset, 'official_split': 'train', 'seed': 20261020,
                    'rng': 'numpy.default_rng.PCG64_class_id_ascending',
                    'train_indices': training, 'validation_indices': validation,
                    'official_class_counts': counts.tolist(), 'validation_class_counts': quotas.tolist(),
                    'train_class_counts': (counts - quotas).tolist(), 'visual_pool_sample_ids': display[:20],
                    'provenance': provenance(), 'official_label_mapping': 'torchvision_no_additional_remapping'}
        atomic_json(ROOT / 'records' / f'{dataset}_split.json', manifest)
        # P0 的独立完整性检查可读取 test 像素和标签范围，不创建 test evaluator。
        test = (CIFAR100(str(ROOT / 'data'), train=False, download=False) if dataset == 'cifar100'
                else SVHN(str(ROOT / 'data/svhn'), split='test', download=False))
        test_images, test_labels = raw_arrays(test, dataset)
        assert len(test_images) == ntest and set(test_labels.tolist()) == set(range(classes))
        lookup = defaultdict(list)
        assigned = np.full(ntrain, 'train', dtype=object)
        assigned[validation] = 'validation'
        for split, array in [('train', images), ('test', test_images)]:
            for index, pixels in enumerate(array):
                digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                lookup[digest].append({'sample_id': f'{dataset}/{split}/{index}',
                                      'assigned_split': str(assigned[index]) if split == 'train' else 'test'})
        duplicates = [{'pixel_sha256': key, 'samples': value} for key, value in lookup.items() if len(value) > 1]
        atomic_json(ROOT / 'records' / f'{dataset}_duplicates.json',
                    {'dataset': dataset, 'groups': duplicates, 'policy': 'audit_only_no_samples_removed'})
        receipt['datasets'][dataset] = {'official_train_n': ntrain, 'train_n': len(training),
              'validation_n': len(validation), 'official_test_n': ntest, 'classes': classes,
              'split_sha256': sha256(ROOT / 'records' / f'{dataset}_split.json'),
              'duplicates_groups': len(duplicates),
              'cross_assigned_split_duplicate_groups': sum(len({p['assigned_split'] for p in d['samples']}) > 1 for d in duplicates),
              'official_test_class_counts': np.bincount(test_labels, minlength=classes).tolist(),
              'split_intersection_empty': True, 'test_evaluator_constructed': False}
        print(json.dumps({'dataset': dataset, **receipt['datasets'][dataset]}), flush=True)
    atomic_json(ROOT / 'records/data_integrity.json', receipt)


if __name__ == '__main__':
    main()
