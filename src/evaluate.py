"""独立的集中测试入口；必须先取得 P3 状态封存和冻结 evaluation manifest。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100, SVHN

from tlcls.common import ROOT, atomic_json, finite_or_raise, load_json, setup, sha256
from tlcls.data import raw_arrays, transform
from tlcls.models import build_model


class TestData(Dataset):
    def __init__(self, dataset, model_key):
        self.dataset, self.model_key = dataset, model_key
        raw = (CIFAR100(str(ROOT / 'data'), train=False, download=False) if dataset == 'cifar100'
               else SVHN(str(ROOT / 'data/svhn'), split='test', download=False))
        self.images, self.labels = raw_arrays(raw, dataset)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample_id = f'{self.dataset}/test/{index}'
        image = transform(Image.fromarray(self.images[index]), self.dataset, self.model_key,
                          False, 0, 0, sample_id)
        return image, int(self.labels[index]), sample_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    assert manifest_path.is_relative_to(ROOT.resolve())
    seal = load_json(ROOT / 'records/P3_sealed.json')
    assert seal['evaluation_manifest_sha256'] == sha256(manifest_path)
    formal_ids = {r['run_id'] for r in load_json(ROOT / 'configs/run_matrix.json')['planned_runs'] if r['phase'] == 'P3'}
    assert set(seal['run_ids']) == formal_ids and len(formal_ids) == 80
    assert seal['selected_hparams_sha256'] == sha256(ROOT / 'selected_hparams.json')
    for run_id in formal_ids:
        status = load_json(ROOT / 'runs' / run_id / 'summary.json')['status']
        assert status not in ['planned', 'running']
    manifest = load_json(manifest_path)
    torch.cuda.set_device(args.device)
    setup(0)
    device = torch.device('cuda', args.device)
    for item in manifest['evaluations']:
        checkpoint = (ROOT / item['checkpoint']).resolve()
        assert checkpoint.is_relative_to(ROOT.resolve()) and sha256(checkpoint) == item['checkpoint_sha256']
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = payload['config']
        model, _ = build_model(config, device)
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        dataset = TestData(config['dataset'], config['model_key'])
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        all_logits, all_labels, ids = [], [], []
        with torch.no_grad():
            for images, labels, sample_ids in dataloader:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(images.to(device, non_blocking=True))
                logits = logits.float()
                finite_or_raise(logits, 'test_logits')
                all_logits.append(logits.cpu())
                all_labels.append(labels)
                ids.extend(sample_ids)
        logits, labels = torch.cat(all_logits), torch.cat(all_labels)
        prediction = logits.argmax(1)
        classes = logits.shape[1]
        confusion = torch.bincount(labels * classes + prediction, minlength=classes * classes).reshape(classes, classes).numpy()
        tp = np.diag(confusion).astype(float)
        recalls = np.divide(tp, confusion.sum(1), out=np.zeros(classes), where=confusion.sum(1) != 0)
        f1 = np.divide(2 * tp, confusion.sum(0) + confusion.sum(1), out=np.zeros(classes),
                       where=(confusion.sum(0) + confusion.sum(1)) != 0)
        name = item['run_id'] + '_' + item['evaluation_role']
        output = ROOT / 'runs' / item['run_id'] / (name + '_predictions.npz')
        np.savez_compressed(output, sample_id=np.asarray(ids), labels=labels.numpy(), prediction=prediction.numpy(), logits=logits.numpy())
        atomic_json(output.with_suffix('.json'), {'run_id': item['run_id'], 'evaluation_role': item['evaluation_role'],
            'checkpoint_sha256': item['checkpoint_sha256'], 'predictions_sha256': sha256(output),
            'n': len(labels), 'top1_percent': float((prediction == labels).float().mean()) * 100,
            'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recalls.mean()),
            'nll': float(torch.nn.functional.cross_entropy(logits, labels)), 'confusion_matrix': confusion.tolist(),
            'manifest_sha256': sha256(manifest_path), 'test_used_for_selection': False})
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
