import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, softmax=True):
        super().__init__()
        self.W1_g = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2_g = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_g = nn.Linear(hidden_size, 1, bias=True)

        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=True)
        self.softmax = softmax

    def forward(self, enc_hid_states, dec_last_hid_state, pointer_mask):

        # Glimpse
        w1_e_g = self.W1_g(enc_hid_states)  # (B, S, H)
        w2_d_g = torch.swapaxes(self.W2_g(dec_last_hid_state), 0, 1)  # (B, 1, H)
        tanh_output_g = torch.tanh(w1_e_g + w2_d_g)  # (B, L, H)
        v_dot_tanh_g = self.V_g(tanh_output_g).squeeze(-1)  # (B, L)
        v_dot_tanh_g = v_dot_tanh_g - 1e10 * (1 - pointer_mask)  # (B, L)

        if self.softmax:
            attention_weights = nn.functional.softmax(v_dot_tanh_g, dim=1)  # (B, L)
        else:
            attention_weights = v_dot_tanh_g

        glimpse = torch.bmm(attention_weights.unsqueeze(1), enc_hid_states).squeeze(
            1
        ) + dec_last_hid_state.squeeze(
            0
        )  # (B, L, H)

        w1_e = self.W1(enc_hid_states)  # (B, L, H)
        w2_d = self.W2(glimpse).unsqueeze(1)  # (Batch_size, 1, hidden_dim)
        tanh_output = torch.tanh(w1_e + w2_d)  # (B, L, H)
        v_dot_tanh = self.V(tanh_output).squeeze(-1)  # (B, L)
        v_dot_tanh = v_dot_tanh - 1e10 * (1 - pointer_mask)  # (B, L)

        if self.softmax:
            attention_weights = nn.functional.softmax(v_dot_tanh, dim=1)
            # Equivalent to log(softmax(x)), but faster and numerically more stable:
            attention_weights_ln = nn.functional.log_softmax(v_dot_tanh, dim=1)
        else:
            attention_weights = v_dot_tanh
            attention_weights_ln = None

        return attention_weights, attention_weights_ln


class CriticNetwork(nn.Module):
    def __init__(self, config):
        super(CriticNetwork, self).__init__()
        self.embedding = nn.Linear(1, config.hid_dim, bias=False)
        self.batch_norm = nn.BatchNorm1d(config.hid_dim)
        self.encoder = nn.LSTM(
            input_size=config.hid_dim, hidden_size=config.hid_dim, batch_first=True
        )
        self.attention = BahdanauAttention(config.hid_dim, softmax=False)
        self.dense1 = nn.Linear(config.max_num_items, config.max_num_items)
        self.dense2 = nn.Linear(config.max_num_items, 1)
        # Constrain initila weights for more stable training
        torch.nn.init.uniform_(self.dense1.weight, -0.08, 0.08)
        torch.nn.init.uniform_(self.dense1.bias, -0.08, 0.08)
        torch.nn.init.uniform_(self.dense2.weight, -0.08, 0.08)
        torch.nn.init.uniform_(self.dense2.bias, -0.08, 0.08)

    def forward(self, states_batch, states_lens, len_mask):

        input_embed = self.embedding(states_batch)
        input_embedded_norm = self.batch_norm(torch.swapaxes(input_embed, 1, 2))
        input_embedded_norm = torch.swapaxes(input_embedded_norm, 1, 2)
        input_embedded_masked = pack_padded_sequence(
            input_embedded_norm, states_lens, batch_first=True, enforce_sorted=False
        )
        enc_output, (h_state, c_state) = self.encoder(input_embedded_masked)
        enc_output = pad_packed_sequence(
            enc_output, batch_first=True, total_length=len_mask.shape[-1]
        )[0]
        enc_last_state = h_state
        att_weights, _ = self.attention(enc_output, enc_last_state, len_mask)
        att_weights = att_weights*len_mask
        x = self.dense1(att_weights)
        pred_reward = self.dense2(x)
        return pred_reward.squeeze(-1)


class ActorPointerNetwork(nn.Module):
    def __init__(self, config):
        super(ActorPointerNetwork, self).__init__()

        self.embedding = nn.Linear(1, config.hid_dim, bias=False)
        self.encoder = nn.LSTM(
            input_size=config.hid_dim, hidden_size=config.hid_dim, batch_first=True
        )
        self.batch_norm = nn.BatchNorm1d(config.hid_dim)
        self.decoder = nn.LSTM(
            input_size=1, hidden_size=config.hid_dim, batch_first=True
        )
        self.attention = BahdanauAttention(config.hid_dim)
        self.max_len = config.max_num_items

        # Initial input to pass to the decoder:
        self.dec_input = -1 * torch.ones(config.batch_size, 1, 1).to(config.device)
        self.device = config.device         

    def forward(self, states_batch, states_lens, len_mask, len_mask_device):

        enc_output, dec_input, h_state, c_state, pointer_mask, actions_seq, actions_log_probs = self.encode_inputs(
            states_batch, states_lens, len_mask, len_mask_device
        )

        for i in range(self.max_len):
            _, (h_state, c_state) = self.decoder(dec_input, (h_state, c_state))
            probs, log_probs = self.attention(enc_output, h_state, pointer_mask)  # (B, L)
            selected_item = torch.multinomial(probs, 1).squeeze(1)  # (batch_size)
            pointer_mask = pointer_mask.scatter_(1, selected_item.unsqueeze(-1), 0)
            log_prob_selected_item = torch.gather(log_probs, 1, selected_item.unsqueeze(-1)).squeeze(1)
            actions_seq[:, i] = selected_item
            actions_log_probs[:, i] = log_prob_selected_item
            dec_input = selected_item.unsqueeze(-1).unsqueeze(-1).to(torch.float32)
            
        
        actions_log_probs = actions_log_probs*len_mask_device
        actions_seq = actions_seq*len_mask - (1 - len_mask)
        return actions_log_probs, actions_seq


    def encode_inputs(self, states_batch, states_lens, len_mask, len_mask_device):
        
        input_embedded = self.embedding(states_batch)  # (batch_size, max_seq_len, hid_dim)
        input_embedded_norm = self.batch_norm(torch.swapaxes(input_embedded, 1, 2))
        input_embedded_norm = torch.swapaxes(input_embedded_norm, 1, 2)  # (B, L, H)
        input_embedded_masked = pack_padded_sequence(
            input_embedded_norm, states_lens, batch_first=True, enforce_sorted=False
        )
        enc_output, (h_state, c_state) = self.encoder(input_embedded_masked)
        enc_output = pad_packed_sequence(
            enc_output, batch_first=True, total_length=len_mask.shape[-1]
        )[0]
        dec_input = self.dec_input
        pointer_mask = len_mask_device.clone()
        actions_seq = -1 * torch.ones_like(len_mask)
        actions_log_probs = torch.zeros_like(len_mask_device, dtype=torch.float32)
        
        return  enc_output, dec_input, h_state, c_state, pointer_mask, actions_seq, actions_log_probs


    @torch.inference_mode()
    def inference(self, states_batch, states_lens, len_mask, len_mask_device):
        
        enc_output, dec_input, h_state, c_state, pointer_mask, actions_seq, _ = self.encode_inputs(
            states_batch, states_lens, len_mask, len_mask_device
        )
        for i in range(self.max_len):
            _, (h_state, c_state) = self.decoder(dec_input, (h_state, c_state))
            probs, _ = self.attention(enc_output, h_state, pointer_mask)  # (B, L)
            selected_item = torch.argmax(probs, axis=1)  # (batch_size)
            pointer_mask = pointer_mask.scatter_(1, selected_item.unsqueeze(-1), 0)
            actions_seq[:, i] = selected_item
            dec_input = selected_item.unsqueeze(-1).unsqueeze(-1).to(torch.float32)            
        
        actions_seq = actions_seq*len_mask - (1 - len_mask)
        return actions_seq