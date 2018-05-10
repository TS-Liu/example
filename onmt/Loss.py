"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import onmt.io


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """
    def __init__(self, generator, generator2, tgt_vocab, tgt_vocab_big):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.generator2 = generator2
        self.tgt_vocab = tgt_vocab
        self.tgt_vocab_big = tgt_vocab_big
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]
        self.unk_idx = tgt_vocab.stoi[onmt.io.UNK]

    def _make_shard_state(self, batch, output, range_, attns=None, attns_2= None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, output2, target_unk, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, output2, attns, attns2):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt[0].size(0))
        shard_state = self._make_shard_state(batch, output, output2, range_, attns, attns2)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, output2, attns, attns_2,
                             cur_trunc, trunc_size, shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, output2, range_, attns, attns_2,)

        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(normalization).backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, LOSS, loss_unk, loss, scores_unk, scores, target_unk, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        pred_unk = scores_unk.max(1)[1]
        non_padding_unk = target_unk.ne(self.padding_idx)
        num_correct_unk = pred_unk.eq(target_unk) \
                          .masked_select(non_padding_unk) \
                          .sum()
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
            .masked_select(non_padding) \
            .sum()
        return onmt.Statistics(LOSS, loss_unk[0], non_padding_unk.sum(), num_correct_unk, loss[0], non_padding.sum(), num_correct)

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, generator, generator2, tgt_vocab, tgt_vocab_big, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, generator2, tgt_vocab, tgt_vocab_big)
        assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)
        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)

            self.criterion2 = nn.KLDivLoss(size_average=False)
            one_hot2 = torch.randn(1, len(tgt_vocab_big))
            one_hot2.fill_(label_smoothing / (len(tgt_vocab_big) - 2))
            one_hot2[0][self.padding_idx] = 0
            self.register_buffer('one_hot2', one_hot2)
        else:
            weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False)
            weight2 = torch.ones(len(tgt_vocab_big))
            weight2[self.padding_idx] = 0
            self.criterion2 = nn.NLLLoss(weight2, size_average=False)
        self.confidence = 1.0 - label_smoothing

    def _make_shard_state(self, batch, output, output2, range_, attns=None, attns_2=None):
        return {
            "output": output,
            "output2": output2,
            "target_unk": batch.tgt[0][range_[0] + 1: range_[1]],
            "target": batch.tgt[1][range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, output2, target_unk, target):
        scores_unk = self.generator(self._bottle(output))
        scores = self.generator2(self._bottle(output2))
        _, vocab_size = scores_unk.size()
        _, vocab_size_big = scores.size()
        tgt_unk_mask = Variable(target_unk.data.eq(self.unk_idx).float().unsqueeze(1)).repeat(1, vocab_size_big, 1).transpose(1,2).contiguous().view(-1,vocab_size_big)  ########
        tgt_no_unk_mask = Variable(target_unk.data.ne(self.unk_idx).float().unsqueeze(1)).repeat(1, vocab_size, 1).transpose(1,2).contiguous().view(-1,vocab_size)
        scores = scores*tgt_unk_mask
        scores_unk = scores_unk*tgt_no_unk_mask

        gtruth_unk = target_unk.view(-1)
        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot2.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)

            tdata_unk = gtruth_unk.data
            mask_unk = torch.nonzero(tdata_unk.eq(self.padding_idx)).squeeze()
            log_likelihood_unk = torch.gather(scores_unk.data, 1, tdata_unk.unsqueeze(1))
            tmp__unk = self.one_hot.repeat(gtruth_unk.size(0), 1)
            tmp__unk.scatter_(1, tdata_unk.unsqueeze(1), self.confidence)
            if mask_unk.dim() > 0:
                log_likelihood_unk.index_fill_(0, mask_unk, 0)
                tmp__unk.index_fill_(0, mask_unk, 0)
            gtruth_unk = Variable(tmp__unk, requires_grad=False)
        loss_unk = self.criterion(scores_unk, gtruth_unk)
        loss = self.criterion2(scores, gtruth)
        # if self.confidence < 1:
        #     # Default: report smoothed ppl.
        #     # loss_data = -log_likelihood.sum(0)
        #     loss_data = loss.data.clone()
        #     loss_data = loss.data.clone()
        # else:
        #     loss_data = loss.data.clone()
        #     loss_data = loss.data.clone()
        loss_data_unk = loss_unk.data.clone()
        loss_data = loss.data.clone()
        LOSS =loss+loss_unk
        target_unk =target_unk * Variable(target_unk.data.ne(self.unk_idx).long())
        target = target * Variable(target_unk.data.eq(self.unk_idx).long())
        stats = self._stats(LOSS,loss_data_unk, loss_data, scores_unk.data, scores.data, target_unk.view(-1).data, target.view(-1).data)

        return LOSS, stats


def filter_shard_state(state, requires_grad=True, volatile=False):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=requires_grad,
                             volatile=volatile)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield filter_shard_state(state, False, True)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
