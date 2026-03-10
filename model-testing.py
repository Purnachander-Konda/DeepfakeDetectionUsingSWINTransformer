import torch
import evaluate
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import numpy as np

feature_extractor = AutoImageProcessor.from_pretrained('./models/swin-tiny-complete')

accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1', average='macro')
precision = evaluate.load('precision', average='macro')
recall = evaluate.load('recall', average='macro')

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    refs = p.label_ids

    results = {}
    results.update(f1.compute(predictions=preds, references=refs, average='macro'))
    results.update(recall.compute(predictions=preds, references=refs, average='macro'))
    results.update(precision.compute(predictions=preds, references=refs, average='macro'))
    results.update(accuracy.compute(predictions=preds, references=refs))

    return results

def transforms(batch):
    inputs = feature_extractor([x.convert('RGB') for x in batch['image']], return_tensors='pt')
    inputs['label'] = batch['label']

    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


test_ds = load_dataset('imagefolder', data_dir='./data', split='test', cache_dir='./cache')
test_ds.set_format(type=test_ds.format['type'], columns=list(test_ds.features.keys()))

processed_test_ds = test_ds.with_transform(transforms)

labels = test_ds.features['label'].names
model = AutoModelForImageClassification.from_pretrained(
    './models/swin-tiny-complete',
    ignore_mismatched_sizes=True,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
)

training_args = TrainingArguments(
    'final-test-results',
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    eval_dataset=processed_test_ds,
    tokenizer=feature_extractor,
)

out = trainer.evaluate()
print(out)