import matplotlib.pyplot as plt
import numpy as np
import re
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
def load_and_preprocess_dataset():
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

    def preprocess_function(examples):
        examples['text'] = [re.sub(r'http\S+', '', text) for text in examples['text']]
        model_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    validation_set = tokenized_datasets['validation']
    split_datasets = validation_set.train_test_split(test_size=0.5)
    split_dataset_dict = DatasetDict({
        'validation': split_datasets['train'],
        'test': split_datasets['test']
    })

    return tokenized_datasets, split_dataset_dict

# Plot label distribution
def plot_label_distribution(dataset):
    labels = dataset['train']['label'] + dataset['validation']['label']
    label_counts = np.bincount(labels)
    labels_unique = sorted(set(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(labels_unique, label_counts[labels_unique], tick_label=[str(label) for label in labels_unique])
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Distribution of Labels in Dataset')
    plt.savefig('Distribution_of_Labels_in_Dataset.png')

# Train model
def train_model(tokenized_datasets, split_dataset_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilroberta-base", num_labels=3).to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        return {'accuracy': acc}

    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=12,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        save_strategy="epoch",
        logging_dir='./logs',
        logging_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=split_dataset_dict['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    train_result = trainer.train()
    return trainer, train_result.metrics, trainer.state.log_history

# Plot training and validation metrics
def plot_metrics(log_history):
    train_losses = []
    eval_losses = []
    eval_accuracies = []
    epochs = []

    for entry in log_history:
        epoch = entry.get('epoch')
        if 'loss' in entry:
            train_losses.append(entry['loss'])
            if epoch not in epochs:
                epochs.append(epoch)
        if 'eval_loss' in entry:
            eval_losses.append(entry['eval_loss'])
        if 'eval_accuracy' in entry:
            eval_accuracies.append(entry['eval_accuracy'])

    plt.figure(figsize=(6, 5))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, eval_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Training_and_Validation_Loss.png')

    plt.figure(figsize=(6, 5))
    plt.plot(epochs, eval_accuracies, 'go-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Validation_Accuracy.png')

# Evaluate model and visualize results
def evaluate_and_visualize(trainer, split_dataset_dict):
    predictions = trainer.predict(split_dataset_dict['test'])
    labels = predictions.label_ids
    preds = np.argmax(predictions.predictions, axis=-1)
    accuracy = accuracy_score(labels, preds)

    print(f"Test Set Accuracy: {accuracy}")

    plt.figure(figsize=(8, 5))
    plt.hist(preds, bins=np.arange(-0.5, 3, 1), alpha=0.7, color='blue', label='Predictions')
    plt.hist(labels, bins=np.arange(-0.5, 3, 1), alpha=0.7, color='red', label='Actual Labels')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predictions and Actual Labels')
    plt.legend()
    plt.grid(True)
    plt.savefig('Distribution_of_Predictions_and_Actual_Labels.png')

    correct_indices = preds == labels
    incorrect_indices = ~correct_indices
    correct_examples = np.random.choice(np.where(correct_indices)[0], size=min(5, sum(correct_indices)), replace=False)
    incorrect_examples = np.random.choice(np.where(incorrect_indices)[0], size=min(5, sum(incorrect_indices)), replace=False)

    print("Correctly Predicted Examples:")
    for i in correct_examples:
        print(f"Text: {split_dataset_dict['test']['text'][i]} - Label: {labels[i]}, Prediction: {preds[i]}")

    print("\nIncorrectly Predicted Examples:")
    for i in incorrect_examples:
        print(f"Text: {split_dataset_dict['test']['text'][i]} - Label: {labels[i]}, Prediction: {preds[i]}")

if __name__ == "__main__":
    print("Loading and preprocessing dataset...")
    tokenized_datasets, split_dataset_dict = load_and_preprocess_dataset()
    print("Plotting label distribution...")
    plot_label_distribution(tokenized_datasets)
    print("Training model...")
    trainer, training_history, log_history = train_model(tokenized_datasets, split_dataset_dict)
    print("Plotting metrics...")
    plot_metrics(log_history)
    print("Evaluating and visualizing results...")
    evaluate_and_visualize(trainer, split_dataset_dict)
