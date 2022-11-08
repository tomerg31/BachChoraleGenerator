import torch
from torch import nn, Tensor
from bag_of_phrase_words import BagOfPhrases, ChoraleSentences
import os


def hot_softmax(y, dim=0, temperature=1.0):
    normalized_y = y - torch.max(y, dim=dim).values
    hot_y = torch.pow(torch.e, torch.div(normalized_y, temperature))
    result = torch.div(hot_y, torch.sum(hot_y, dim=dim))

    return result


def generate_from_model(model, chorale_sentences: ChoraleSentences, T, pick_first_phrase=False):
    device = next(model.parameters()).device

    if pick_first_phrase:
        new_input_index = chorale_sentences.get_starting_phrase()
    else:
        new_input_index = chorale_sentences.start_token_index

    generated_chorale_sentence = [new_input_index]
    model_input = torch.unsqueeze(
        chorale_sentences.phrase_index_to_onehot(generated_chorale_sentence).to(device).float(),
        dim=0
    )

    with torch.no_grad():
        hidden_state = None
        counter = 0
        while new_input_index != chorale_sentences.end_token_index and counter < 25:
            model_output, hidden_state = model(model_input, hidden_state)
            last_char_distribution = hot_softmax(model_output.squeeze(), dim=0, temperature=T)
            new_input_index = torch.multinomial(last_char_distribution, 1).item()
            generated_chorale_sentence.append(new_input_index)
            model_input = torch.unsqueeze(
                chorale_sentences.phrase_index_to_onehot(new_input_index).to(device).float(),
                dim=0
            )
            counter += 1

    return generated_chorale_sentence


class MySequenceModel(nn.Module):
    def __init__(self, seq_model, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.initial_hidden_state = nn.Parameter(torch.zeros(self.n_layers, 1, self.h_dim))

        self.seq_model = seq_model(
            input_size=in_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=dropout
        )

        self.out_layer = nn.Linear(self.h_dim, self.out_dim)

        self.register_parameter('initial_hidden_state', self.initial_hidden_state)
        self.add_module('seq_model', self.seq_model)
        self.add_module('out_layer', self.out_layer)

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        batch_size, seq_len, _ = input.shape

        if hidden_state is None:
            hidden_state = torch.concat([self.initial_hidden_state for _ in range(batch_size)], dim=1)

        seq_model_output, output_hidden_state = self.seq_model(input, hidden_state)

        return self.out_layer(seq_model_output), output_hidden_state
