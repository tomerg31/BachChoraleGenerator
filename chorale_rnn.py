import torch
from torch import nn, Tensor
from bag_of_phrase_words import BagOfPhrases, ChoraleSentences
from trainer import RNNTrainer
import os


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
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
        self.layer_params = []
        self.dropout = dropout

        self.initial_hidden_state = nn.Parameter(torch.zeros(1, self.n_layers, self.h_dim))

        for i in range(self.n_layers):
            if i == 0:
                W_xz = nn.Linear(self.in_dim, self.h_dim)
                W_xr = nn.Linear(self.in_dim, self.h_dim)
                W_xg = nn.Linear(self.in_dim, self.h_dim)
            else:
                W_xz = nn.Linear(self.h_dim, self.h_dim)
                W_xr = nn.Linear(self.h_dim, self.h_dim)
                W_xg = nn.Linear(self.h_dim, self.h_dim)
            W_hz = nn.Linear(self.h_dim, self.h_dim, bias=False)
            W_hr = nn.Linear(self.h_dim, self.h_dim, bias=False)
            W_hg = nn.Linear(self.h_dim, self.h_dim, bias=False)

            self.layer_params += [
                {
                    'W_xz': W_xz,
                    'W_xr': W_xr,
                    'W_xg': W_xg,
                    'W_hz': W_hz,
                    'W_hr': W_hr,
                    'W_hg': W_hg,
                }
            ]
        W_hy = nn.Linear(self.h_dim, self.out_dim)
        self.layer_params += [{'W_hy': W_hy}]
        for i, layer in enumerate(self.layer_params):
            for key in layer.keys():
                self.add_module(str(i) + ' ' + key, self.layer_params[i][key])

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            # If hidden state is None, then we are generating a phrase from the beginning, we want the learned initial
            # hidden state
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(1, self.h_dim)  # self.initial_hidden_state[:, i, :]
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        sig = nn.Sigmoid()
        tanh = nn.Tanh()
        dropout = nn.Dropout(self.dropout) if self.dropout else nn.Identity()

        for layer_index, layer in enumerate(self.layer_params[:-1]):
            outputs = []
            for char_index in range(input.size(1)):
                z_t = sig(
                    layer['W_xz'](layer_input[:, char_index, :]) +
                    layer['W_hz'](layer_states[layer_index])
                )

                r_t = sig(
                    layer['W_xr'](layer_input[:, char_index, :]) +
                    layer['W_hr'](layer_states[layer_index])
                )

                g_t = tanh(
                    layer['W_xg'](layer_input[:, char_index, :]) +
                    layer['W_hg'](r_t * layer_states[layer_index])
                )

                layer_states[layer_index] = z_t * layer_states[layer_index] + (1 - z_t) * g_t
                outputs += [layer_states[layer_index]]

            layer_input = dropout(torch.stack([*outputs], dim=1))

        layer_output = self.layer_params[-1]['W_hy'](layer_input)
        hidden_state = torch.stack([*layer_states], dim=1)

        return layer_output, hidden_state


def chorale_sentences_to_labelled_samples(chorale_sentences, device):
    embedded_chorale_sentences = torch.nn.functional.one_hot(chorale_sentences.chorale_sentence_tensor).to(device)

    samples = embedded_chorale_sentences[:, :-1, :]
    labels = chorale_sentences.chorale_sentence_tensor[:, 1:].to(device)

    samples = [
        torch.unsqueeze(torch.nn.functional.one_hot(chorale_sentence).to(device)[:-1], dim=0) for chorale_sentence in
        chorale_sentences.chorale_sentence_list_of_tensors
    ]

    labels = [
        torch.unsqueeze(chorale_sentence[1:], dim=0) for chorale_sentence in
        chorale_sentences.chorale_sentence_list_of_tensors
    ]

    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    normalized_y = y - torch.max(y, dim=dim).values
    hot_y = torch.pow(torch.e, torch.div(normalized_y, temperature))
    result = torch.div(hot_y, torch.sum(hot_y, dim=dim))

    return result


def generate_from_model(model, chorale_sentences: ChoraleSentences, T):
    device = next(model.parameters()).device
    generated_chorale_sentence = [chorale_sentences.start_token_index]
    model_input = torch.unsqueeze(
        chorale_sentences.phrase_index_to_onehot(generated_chorale_sentence).to(device).float(),
        dim=0
    )

    with torch.no_grad():
        hidden_state = None
        for _ in range(chorale_sentences.max_chorale_length):
            model_output, hidden_state = model(model_input, hidden_state)
            last_char_distribution = hot_softmax(model_output.squeeze(), dim=0, temperature=T)
            new_input_index = torch.multinomial(last_char_distribution, 1).item()
            generated_chorale_sentence.append(new_input_index)
            model_input = torch.unsqueeze(
                chorale_sentences.phrase_index_to_onehot(new_input_index).to(device).float(),
                dim=0
            )

    return generated_chorale_sentence


def train_rnn(chorale_sentences: ChoraleSentences, hyperparams: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    print('hyperparams:\n', hyperparams)

    # Dataset definition
    vocab_len = chorale_sentences.vocabulary_length

    # Since we have relatively few chorales, and each is a different length, we will insert each one to the network
    # individually.
    batch_size = hyperparams['batch_size']

    train_test_ratio = 0.9
    num_samples = chorale_sentences.number_of_chorales
    num_train = int(train_test_ratio * num_samples)

    samples, labels = chorale_sentences_to_labelled_samples(chorale_sentences, device)

    ds_train = torch.utils.data.TensorDataset(*samples[:1], *labels[:1])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False, drop_last=True)

    ds_test = torch.utils.data.TensorDataset(*samples[:1], *labels[:1])
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=False, drop_last=True)

    print(f'Train: {len(dl_train):3d} batches,')
    print(f'Test:  {len(dl_test):3d} batches,')

    # Training definition
    in_dim = out_dim = vocab_len
    checkpoint_file = 'checkpoints/rnn'
    num_epochs = 50
    early_stopping = hyperparams['epochs_wo_improvement']

    model = MultilayerGRU(
        in_dim,
        hyperparams['h_dim'],
        out_dim,
        hyperparams['n_layers'],
        hyperparams['dropout']
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learn_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=hyperparams['lr_sched_factor'],
        patience=hyperparams['lr_sched_patience'],
        verbose=True
    )

    print(model)

    trainer = RNNTrainer(model, loss_fn, optimizer, device)

    def post_epoch_fn(epoch, train_res, test_res, verbose):
        # Update learning rate
        scheduler.step(test_res.accuracy)
        # Sample from model to show progress
        if verbose:
            generated_sequence = generate_from_model(
                model, chorale_sentences, T=0.5
            )
            print(generated_sequence)

    # Train, unless final checkpoint is found
    checkpoint_file_final = f'{checkpoint_file}_final.pt'
    if os.path.isfile(checkpoint_file_final):
        print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
        saved_state = torch.load(checkpoint_file_final, map_location=device)
        model.load_state_dict(saved_state['model_state'])
    else:
        try:
            # Print pre-training sampling
            print(generate_from_model(model, chorale_sentences, T=0.5))

            fit_res = trainer.fit(dl_train, dl_test, num_epochs, max_batches=None,
                                  post_epoch_fn=post_epoch_fn, early_stopping=early_stopping,
                                  checkpoints=checkpoint_file, print_every=1)

        except KeyboardInterrupt as e:
            print('\n *** Training interrupted by user')

    return model

