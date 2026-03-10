import torch
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoFeatureExtractor, SwinForImageClassification, Trainer, TrainingArguments
import evaluate
from torchsummary import summary

# Creating global variables as they will instead have to be created multiple times within functions.
feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
f1 = evaluate.load("f1", average='macro')
prec = evaluate.load("precision", average='macro')
recall = evaluate.load("recall", average='macro')
accuracy = evaluate.load("accuracy")


def compute_metrics(p):
    """
    This function is passed as a reference to the Trainer object. It calculates four metrics: f1 score, precision,
    recall, and accuracy. The metrics are saved in a dict object which is then returned.


    :param p: the predictions object containing two attributes for predictions made for a batch, and their respective
    true label ids.

    :return: a dict containing the results of four metrics: f1 score, precision, recall, and accuracy.
    """
    preds = np.argmax(p.predictions, axis=1)
    refs = p.label_ids

    results = {}
    results.update(f1.compute(predictions=preds, references=refs, average='macro'))
    results.update(prec.compute(predictions=preds, references=refs, average='macro'))
    results.update(recall.compute(predictions=preds, references=refs, average='macro'))
    results.update(accuracy.compute(predictions=preds, references=refs))
    return results


def transforms(batch):
    """
    This function performs the required transformations batch-by-batch to images before passing them to the model for
    training / testing. It ensures that the images are of size 224*224 and arranged in a torch-compatible tensor format.


    :param batch:  The batch of images to be transformed.

    :return: A pytorch tensor containing transformed batch of images.
    """
    inputs = feature_extractor([x.convert('RGB') for x in batch['image']], return_tensors='pt')
    inputs['label'] = batch['label']
    return inputs


def collate_fn(batch):
    """
    It collates the images by stacking the batch of images on top of each other to make training process faster.

    :param batch: Batch of images to be collated.

    :return: A dictionary containing a tensor of images stacked on top of each other, and their respective labels.
    """
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


# Loading the dataset from the ./data folder
ds = load_dataset("imagefolder", data_dir="./data", cache_dir='./cache')

# Pre-processing the dataset
processed_ds = ds.with_transform(transforms)

labels = ds['train'].features['label'].names

# Loading the SWIN Model from local storage, if a model is saved. Otherwise, pre-trained weights are loaded.
model = SwinForImageClassification.from_pretrained(
    'microsoft/swin-tiny-patch4-window7-224' if not os.path.exists('./models/swin-tiny-complete')
    else './models/swin-tiny-complete',
    ignore_mismatched_sizes=True,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
)
print('Model Summary:\n' + str(summary(model, (3, 224, 224))))

# Common batch size for training and evaluation.
batch_size = 4

# Creating the TrainingArguments object to be passed to the trainer object.
training_args = TrainingArguments(
    f'./results/swin-tiny-complete',
    remove_unused_columns=False,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    warmup_ratio=0.01,
    weight_decay=0.01,
    logging_steps=4000,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    push_to_hub=False,
)

# Trainer object containing model, arguments, along with collate, metrics functions and feature extractor is created.
# The datasets are passed separately for training and evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=processed_ds['train'],
    eval_dataset=processed_ds['test'],
    tokenizer=feature_extractor,
)

# Training the model and saving all results.
train_results = trainer.train()
print('Training Done!')

# Saving the model
trainer.save_model(output_dir='./models/swin-tiny-complete')
print('Saved Model')

# Calculating and saving training metrics.
trainer.log_metrics('train_results', train_results.metrics)
trainer.save_metrics('train_results', train_results.metrics)
trainer.save_state()
print('Saved Training Metrics!')

# Calculating and saving testing metrics.
metrics = trainer.evaluate(processed_ds['test'])
trainer.log_metrics('test_results', metrics)
trainer.save_metrics('test_results', metrics)
print('Saved Testing Metrics!')
