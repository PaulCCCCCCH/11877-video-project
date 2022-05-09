import torch
from torch import nn

# from https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html
class LockedDropout(nn.Module):
    """ LockedDropout applies the same dropout mask to every time step.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'

def get_last_element(tensor, lengths):
  """tensor shape: T, B, C"""
  feature_dim_size = tensor.size(2)
  last_element_selector = (lengths - 1).unsqueeze(1).repeat(1, feature_dim_size).unsqueeze(0).to(tensor.device)
  last_elements = torch.gather(tensor, 0, last_element_selector).squeeze(0)
  return last_elements

class LSTMLanguageModel(nn.Module):
  def __init__(self, vocab_size=49408, embedding_size=400, hidden_size=1500):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.rnn = nn.Sequential(
      nn.LSTM(embedding_size, hidden_size),
      LockedDropout(0.2),
      nn.LSTM(hidden_size, hidden_size),
      LockedDropout(0.2),
      nn.LSTM(hidden_size, embedding_size)
    )
    
    self.word_prob = nn.Identity()

#     self.word_prob = nn.Linear(embedding_size, vocab_size)

#     # weight tying
#     self.word_prob.weight = self.embedding.weight

  def forward(self, inputs, init_states=None):
    # Feel free to add extra arguments to forward (like an argument to pass in the hiddens)
    inputs = inputs.permute(1, 0) # switch to batch second
    embed = self.embedding(inputs)

    rnn_data = embed

    rnn_layer_index = 0
    new_states = list()

    for layer in self.rnn:
      if type(layer) is nn.LSTM:
        if init_states is None:
          layer_init_state = None
        else:
          layer_init_state = init_states[rnn_layer_index]

        rnn_data, last_states = layer(rnn_data, layer_init_state)
        new_states.append(last_states)
        rnn_layer_index += 1
      else:
        rnn_data = layer(rnn_data)

    return self.word_prob(rnn_data), new_states