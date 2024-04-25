import json
from random import shuffle
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, evaluation

def train_model(train_examples, output_path, backcbone_model = 'all-MiniLM-L6-v2',
                epochs=1, batch_size=16, evaluation_steps=1000, validation_size=1000,learning_rate=2e-5):
    """
    Train a SentenceTransformer model with provided training examples.

    Args:
    train_examples (list): A list of training examples.
    output_path (str): Directory where the model will be saved.
    epochs (int): Number of training epochs.
    batch_size (int): Size of each training batch.
    evaluation_steps (int): Steps interval to perform evaluation.
    validation_size (int): Number of examples to use for validation.
    """
    shuffle(train_examples)  # Shuffle the dataset before splitting

    # Splitting the dataset into training and validation
    validation_examples = train_examples[:validation_size]
    train_examples = train_examples[validation_size:]

    # Convert datasets to DataLoaders
    train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)

    # Load the SentenceTransformer model
    model = SentenceTransformer(backcbone_model)
    
    # Define the loss function
    train_loss = losses.CosineSimilarityLoss(model=model)

    # Define an evaluator using the validation dataset
    dev_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(validation_examples, name='dev')

    optimizer_params = {'lr': learning_rate, 'eps': 1e-6}

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        evaluator=dev_evaluator,
        evaluation_steps=evaluation_steps,
        optimizer_params=optimizer_params,
        output_path=output_path
    )


if __name__ == '__main__':
    # Configuration parameters
    json_path = 'ts_sample_train_examples.json'
    output_path = './output/finetuned_model'
    epochs = 1
    evaluation_steps = 1000
    validation_size = 1000

    # Load training examples from JSON file
    with open(json_path, 'r') as f:
        train_examples = json.load(f)

    train_model(train_examples, output_path, epochs=epochs, evaluation_steps=evaluation_steps, validation_size=validation_size)
