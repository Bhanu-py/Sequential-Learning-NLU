from transformers import BertForSequenceClassification


def pretrained(num_labels=7, model="bert-base-uncased"):
    model = BertForSequenceClassification.from_pretrained(
        model,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--7for Multi-class classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    return model




if __name__ == '__main__':
    # Get all the model's parameters as a list of tuples.
    params = list(pretrained().named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
