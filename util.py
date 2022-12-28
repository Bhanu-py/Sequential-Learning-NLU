from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
import torch
import numpy as np

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def pre_process(sentences, MAX_LEN=80):
    # Tokenize all the sentences and map the tokens to thier word IDs.
    input_ids = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        )

        input_ids.append(encoded_sent)
    # print(input_ids)

    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                              dtype="long", truncating="post", padding="post")
    # input_ids

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)

    return prediction_inputs, prediction_masks


def predict(sentences, model, MAX_LEN=80):
    # Prediction on test set

    # print('Predicting labels for {:,} test sentences...'.format(len(sentences)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions = []

    # Predict
    for sent in sentences:
        # Add batch to GPU
        # batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = sent
        print(b_input_ids)
        print(b_input_mask)
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        # true_labels.append(label_ids)
    print('    DONE.')
    return predictions


def softmax(z):
  '''Return the softmax output of a vector.'''
  exp_z = np.exp(z)
  sum = exp_z.sum()
  softmax_z = np.round(exp_z/sum,3)
  return softmax_z


def flat_accuracy_batch(logits, labels):
  regular_list = np.argmax(logits, axis=1)
  pred = regular_list
  regular_label_list = np.argmax(labels, axis=1)
  labels_flat = regular_label_list
  return np.mean(np.array(pred) == np.array(labels_flat))


if __name__ == '__main__':
    pre_process()