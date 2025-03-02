import tensorflow as tf
from tensorflow.python.util import nest
from tpolicies.utils.distributions import MaskSeqCategoricalPd
from tpolicies.utils.sequence_ops import multistep_forward_view
from tpolicies.utils.vtrace_ops import vtrace_from_importance_weights


import keras
from keras import layers

def multi_head_xe_loss(inputs_action_logits,
                       inputs_action_labels,
                       inputs_mask_weights):
  """"Multi-head cross entropy loss.

  Args:
    inputs_action_logits: logits, organized in a nest.map_structure compatible
      structure of Tensors.
    inputs_action_labels:
    inputs_mask_weights:

  Returns:
    A Tensor, total loss.
    A structure of Tensor, per-head loss. The same structure as inputs.
  """
  def _each_xe_loss(a_logits, a_label, weight):
    # make weight broadcast-able
    while a_label.shape.rank > weight.shape.rank:
      weight = tf.expand_dims(weight, axis=-1)
    if a_logits.shape.rank == a_label.shape.rank:
      # deemed as MultiBinary (multi label, each is zero/one)
      # e.g., a_label: (bs, 600), a_logits: (bs, 600)
      loss = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=a_label, logits=a_logits, weights=weight,
        reduction=tf.compat.v1.losses.Reduction.NONE  # keep the batch_size dim
      )
    else:
      # deemed as Discrete (mutually exclusive multi-class)
      # e.g., a_label: (bs, d1,..., dM), a_logits: (bs, d1,..., dM, K)
      loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=a_label, logits=a_logits, weights=weight,
        reduction=tf.compat.v1.losses.Reduction.NONE  # keep the batch_size dim
      )
    # make sure the loss in shape (bs,)
    while loss.shape.rank > 1:
      loss = tf.reduce_sum(loss, axis=-1)
    return loss

  head_xe_loss = nest.map_structure(_each_xe_loss, inputs_action_logits,
                                    inputs_action_labels, inputs_mask_weights)

  final_xe_loss = tf.add_n(nest.flatten(head_xe_loss))
  return final_xe_loss, head_xe_loss


def multi_head_neglogp_loss(inputs_action_pds,
                            inputs_action_labels,
                            inputs_mask_weights,
                            set_loss=False):
  """"Multi-head neglogp loss.

  Args:
    inputs_action_pds: pds, organized in a nest.map_structure compatible
      structure of Tensors.
    inputs_action_labels:
    inputs_mask_weights:

  Returns:
    A Tensor, total loss.
    A structure of Tensor, per-head loss. The same structure as inputs.
  """
  def _each_neglogp_loss(a_pd, a_label, weight):
    # make weight broadcast-able
    while (a_label.shape.rank > weight.shape.rank
           + isinstance(a_pd, MaskSeqCategoricalPd)):
      weight = tf.expand_dims(weight, axis=-1)
    if isinstance(a_pd, MaskSeqCategoricalPd):
      if set_loss:
        loss = a_pd.set_xe(a_label, mean=True) * weight
      else:
        loss = a_pd.neglogp(a_label, mean=True) * weight
    else:
      loss = a_pd.neglogp(a_label) * weight
    # make sure the loss in shape (bs,)
    while loss.shape.rank > 1:
      loss = tf.reduce_sum(loss, axis=-1)
    return loss

  head_neglogp_loss = nest.map_structure(
    _each_neglogp_loss, inputs_action_pds,
    inputs_action_labels, inputs_mask_weights)

  final_neglogp_loss = tf.add_n(nest.flatten(head_neglogp_loss))
  return final_neglogp_loss, head_neglogp_loss


def distill_loss_old(student_logits, teacher_logits, masks):
  """ Distillation loss.

    Args:
      student_logits: structured logits compatible with nest.map_structure.
      teacher_logits:

    Returns:
      final_distill_loss: total loss.
      head_distill_loss: per-head loss. The same structure as inputs.
  """
  def _compute_kl(logits, o_logits, masks):
    a0 = logits - tf.reduce_max(logits, axis=-1, keep_dims=True)
    a1 = o_logits - tf.reduce_max(o_logits, axis=-1, keep_dims=True)
    ea0 = tf.exp(a0)
    ea1 = tf.exp(a1)
    z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
    z1 = tf.reduce_sum(ea1, axis=-1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1) * masks

  head_distill_loss = nest.map_structure(_compute_kl, student_logits, teacher_logits, masks)
  return head_distill_loss


def distill_loss(student_pds, teacher_flatparam, masks=None):
  """ Distillation loss.
    Args:
      student_pds: structured pds compatible with nest.map_structure.
      teacher_flatparam: logits for categorical and
      concat([mean, var]) for DiagGaussian

    Returns:
      final_distill_loss: total loss.
      head_distill_loss: per-head loss. The same structure as inputs.
  """
  from tpolicies.utils.distributions import CategoricalPd
  from tpolicies.utils.distributions import DiagGaussianPd
  class FakeCategoricalPD(object):
    def __init__(self, logits):
      self.logits = logits
  class FakeDiagGaussianPD(object):
    def __init__(self, mean, logstd):
      self.mean = mean
      self.logstd = logstd
      self.std = tf.exp(logstd)
  def _make_fake_pd(pd, o_flatparam):
    if isinstance(pd, CategoricalPd):
      o = FakeCategoricalPD(o_flatparam)
    elif isinstance(pd, DiagGaussianPd):
      o_mean, o_logstd = tf.split(
        axis=-1, num_or_size_splits=2, value=o_flatparam)
      o = FakeDiagGaussianPD(o_mean, o_logstd)
    else:
      raise NotImplementedError("Unknown PD type {}.".format(pd))
    return o
  def _compute_kl_with_mask(pd, o_flatparam, mask):
    o = _make_fake_pd(pd, o_flatparam)
    return tf.reduce_mean(pd.kl(o) * mask)
  def _compute_kl(pd, o_flatparam):
    o = _make_fake_pd(pd, o_flatparam)
    return tf.reduce_mean(pd.kl(o))

  if masks is not None:
    head_distill_loss = nest.map_structure(
      _compute_kl_with_mask, student_pds, teacher_flatparam, masks)
  else:
    head_distill_loss = nest.map_structure(
      _compute_kl, student_pds, teacher_flatparam)
  return head_distill_loss


def distill_loss_v2(student_pds, teacher_mean, teacher_logstd, masks):
  """ Distillation loss for continuous case.
    Args:
      student_pds: structured pds compatible with nest.map_structure.
      teacher_mean:
      teacher_logstd:

    Returns:
      final_distill_loss: total loss.
      head_distill_loss: per-head loss. The same structure as inputs.
  """
  class PD(object):
    def __init__(self, mean, logstd):
      self.mean = mean
      self.logstd = logstd
      self.std = 2.718281828459045 ** logstd
  def _compute_kl(pd, o_mean, o_logstd, mask):
    o = PD(o_mean, o_logstd)
    return tf.reduce_mean(pd.kl(o) * mask)

  head_distill_loss = nest.map_structure(
    _compute_kl, student_pds, teacher_mean, teacher_logstd, masks)
  return head_distill_loss


def rgps_loss(student_pd, teacher_logits, mask):
  a0 = student_pd.logits - tf.reduce_max(student_pd.logits, axis=-1, keep_dims=True)
  a1 = teacher_logits - tf.reduce_max(teacher_logits, axis=-1, keep_dims=True)
  ea0 = tf.exp(a0)
  ea1 = tf.exp(a1)
  z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
  z1 = tf.reduce_sum(ea1, axis=-1, keep_dims=True)
  # p0 = ea0 / z0
  p1 = ea1 / z1
  loss = tf.squeeze(tf.cast(mask, dtype=tf.float32)) * \
         tf.reduce_sum(p1 * (tf.log(z0) - a0), axis=-1)
  return tf.reduce_mean(loss)


def ppo_loss(neglogp, oldneglogp, vpred, R, V, masks=None, reward_weights=None,
             merge_pi=True, adv_normalize=True, clip_range=0.1,
             clip_range_lower=0.1, sync_statistics=None,
             batch_sam_weight=None):
  """"PPO loss.

  Not recommended, use `ppo2_loss` instead. Use it only for backwards
  compatibility. This ppo loss impl
  * can handle the structured action heads that it sums (NOT mean) the pg loss
    over each action head
  * couples the pg loss and value loss, returning them both
  which is slightly over-engineering when being convenient to call.

  Args:
    neglogp: neglogp, structure as ac_space.
    oldneglogp: neglogp of pi_old, same structure with neglogp.
    vpred: predicted v, in shape [batch_size, n_v]
    R: return from actor, in shape [batch_size, n_v]
    V: value from actor, in shape [batch_size, n_v]
    masks: action logits mask in 0/1 value. The same structure and shape with
      neglogp
    reward_weights: reward weights in shape [1, n_v] (merge_pi=True or False)
      or [len(neglogp), n_v] (MUST merge_pi=False)
    merge_pi: whether to merge pi, if True, original PPO, else split PPO
      (shared adv or independent adv decided by the shape of reward_weights)
    adv_normalize: if normalize advantage
    clip_range: clip range
    sync_statistics: if synchronize statistics across multi GPUs (if any)

  Returns:
    pg_loss: policy loss, a scalar in shape []
    vf_loss: value loss, a Tensor in shape [n_v,]
  """
  nest.assert_same_structure(neglogp, oldneglogp)
  ratio = (tf.stack(nest.flatten(oldneglogp), axis=1) -
           tf.stack(nest.flatten(neglogp), axis=1))
  if masks is not None:
    nest.assert_same_structure(neglogp, masks)
    ratio = tf.stack(nest.flatten(masks), axis=1) * ratio
  if merge_pi:
    ratio = tf.exp(tf.reduce_sum(ratio, axis=-1, keepdims=True))
  else:
    ratio = tf.exp(ratio)

  # normalize ADV
  adv = R - V
  if reward_weights is not None:
    adv = tf.matmul(adv, reward_weights, transpose_b=True)
  batch_mean = tf.reduce_mean(adv, axis=0)
  batch_mean_square = tf.reduce_mean(tf.square(adv), axis=0)
  if sync_statistics == 'horovod':
    # https://github.com/tensorpack/tensorpack/blob/07783edb998cec3ec91c4312b39bd754cf9ececa/tensorpack/models/batch_norm.py#L226-L231
    import horovod.tensorflow as hvd
    batch_mean = hvd.allreduce(batch_mean, average=True)
    batch_mean_square = hvd.allreduce(batch_mean_square, average=True)
  batch_std = tf.sqrt(batch_mean_square - tf.square(batch_mean))
  if adv_normalize:
    adv = (adv - batch_mean) / (batch_std + 1e-8)

  vpredclipped = V + tf.clip_by_value(vpred - V, - clip_range, clip_range)
  vf_losses1 = tf.square(vpred - R)
  vf_losses2 = tf.square(vpredclipped - R)
  # TODO: add sample weight here. also pg_loss, distill_loos, entropy
  if batch_sam_weight is not None:
    vf_loss = .5 * tf.reduce_sum(tf.maximum(vf_losses1, vf_losses2) * batch_sam_weight,
                                 axis=0) / (tf.reduce_sum(batch_sam_weight) + 1e-8)
  else:
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2), axis=0)
  pg_losses1 = -adv * ratio
  pg_losses2 = -adv * tf.clip_by_value(ratio, 1.0 - clip_range,
                                       1.0 + clip_range)
  pg_losses3 = -adv * tf.clip_by_value(ratio, 1.0 - clip_range,
                                       1.0 + clip_range_lower)
  ppo_loss_final = tf.where(tf.greater(tf.tile(adv, [1, ratio.shape[-1]]), 0),
                            tf.maximum(pg_losses1, pg_losses2), pg_losses3)  # (batch, head_dim)
  if batch_sam_weight is not None:
    ppo_loss_final = ppo_loss_final * batch_sam_weight
    pg_loss = tf.reduce_sum(tf.reduce_sum(ppo_loss_final, axis=0) / (tf.reduce_sum(batch_sam_weight)) + 1e-8)
  else:
    pg_loss = tf.reduce_sum(tf.reduce_mean(ppo_loss_final, axis=0))
  return pg_loss, vf_loss


def ppo2_loss(neglogp, oldneglogp, vpred, R, mask=None, adv_normalize=True,
              clip_range=0.1, clip_range_lower=0.1, sync_statistics=None):
  """"PPO2 loss.

  The PPO implementation where the values are computed at the learner's end,
  i.e., there is no `oldvpred` term. This treatment is suitable for PPO with
  large Replay Memory where the off-policy effect bulks.

  It accepts the shape layout (b1, b2, ...) which will be taken as the
  batch_size dimension. For example, the neglopg
  * can be (batch_size,)
  * can be (T, B) where T=rollout_len, B=n_rollout

  Args:
    neglogp: neglogp of pi, in shape (b1, b2,...)
    oldneglogp: neglogp of pi_old, in shape (b1, b2,...)
    vpred: predicted v, in shape (b1, b2, ...)
    R: (lambda) return, in shape (b1, b2, ...)
    mask: action logits mask in 0/1 value, in shape (b1, b2,...), the
      same shape with `neglogp` and `oldneglogp`
    adv_normalize: whether to normalize advantage
    clip_range: clip range, scalar
    sync_statistics: whether to synchronize statistics across multi GPUs (if
      any)

  Returns:
    pg_loss: the PPO policy loss, a scalar in shape ()
  """
  # Note the negative sign; we want ratio = pi / pi_{old}
  ratio = oldneglogp - neglogp
  if mask is not None:
    ratio = mask * ratio
  ratio = tf.exp(ratio)

  # make the advantage
  # Note: DO the stop_gradient stuff AT THE CALLER'S END when necessary
  # R = tf.stop_gradient(R)
  # vpred = tf.stop_gradient(vpred)
  adv = R - vpred
  # normalize the advantage
  batch_mean = tf.reduce_mean(adv)
  batch_mean_square = tf.reduce_mean(tf.square(adv))
  if sync_statistics == 'horovod':
    # https://github.com/tensorpack/tensorpack/blob/07783edb998cec3ec91c4312b39bd754cf9ececa/tensorpack/models/batch_norm.py#L226-L231
    import horovod.tensorflow as hvd
    batch_mean = hvd.allreduce(batch_mean, average=True)
    batch_mean_square = hvd.allreduce(batch_mean_square, average=True)
  batch_std = tf.sqrt(batch_mean_square - tf.square(batch_mean))
  if adv_normalize:
    adv = (adv - batch_mean) / (batch_std + 1e-8)

  # the ppo policy gradient loss
  pg_losses1 = -adv * ratio
  pg_losses2 = -adv * tf.clip_by_value(ratio, 1.0 - clip_range,
                                       1.0 + clip_range)
  pg_losses3 = -adv * tf.clip_by_value(ratio, 1.0 - clip_range,
                                       1.0 + clip_range_lower)
  # pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))  ## the original PPO
  # dual clip PPO
  pg_loss = tf.reduce_mean(
    tf.where(tf.greater(adv, 0), tf.maximum(pg_losses1, pg_losses2), pg_losses3))
  return pg_loss


def td_lambda(values, rewards, discounts, lam):
  """ td_lambda_loss for values

    Args:
      values: predicted v, in shape [T, B]
      rewards: rewards from actor, in shape [T, B]
      discounts: discounts from actor, in shape [T, B]
      lam: lambda, scalar

    Returns:
      vf_loss: value loss, a list of Tensor of len n_v
  """
  with tf.device("/cpu:0"):
    lam_return = multistep_forward_view(rewards[:-1], discounts[:-1],
                                        values[1:], lam)
  loss = tf.reduce_mean(0.5 * tf.square(
    tf.stop_gradient(lam_return) - values[:-1]))
  return loss


def upgo_loss(neglogp, oldneglogp, mask, values, rewards, discounts,
              clip_pg_rho_threshold=1.0):
  """ upgo_returns for values

      Args:
        neglogp: -log(pi), in shape [T, B, n_acts]
        oldneglogp: -log(pi_old), in shape [T, B, n_acts]
        mask: arg mask, in shape [T, B, n_acts]
        values: predicted v, in shape [T, B]
        rewards: rewards from actor, in shape [T, B]
        discounts: discounts from actor, in shape [T, B]

      Returns:
        loss: upgo loss,  a list of Tensor of len n_v
    """
  lam = tf.cast(rewards[:-1] + discounts[:-1] * values[1:] >= values[:-1],
                dtype=tf.float32)
  lam = tf.concat([lam[1:], tf.zeros_like(lam[-1:])], axis=0)
  with tf.device("/cpu:0"):
    lambda_return = multistep_forward_view(rewards[:-1], discounts[:-1],
                                           values[1:], lam)
  ratio = tf.exp(oldneglogp - neglogp)
  if clip_pg_rho_threshold is not None:
    clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, ratio,
                                 name='clipped_pg_rhos')
  else:
    clipped_pg_rhos = ratio
  adv = tf.expand_dims(lambda_return - values[:-1], -1) * clipped_pg_rhos[:-1]
  if mask is None:
    loss = tf.reduce_mean(tf.stop_gradient(adv) * neglogp[:-1])
  else:
    loss = tf.reduce_mean(mask[:-1] * tf.stop_gradient(adv) * neglogp[:-1])
  return loss


def vtrace_loss(neglogp, oldneglogp, mask, values, rewards, discounts,
                clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
  """Vtrace Loss

  Args:
    neglogp: -log(pi), in shape [T, B, n_acts]
    oldneglogp: -log(pi_old), in shape [T, B, n_acts]
    mask: arg mask, in shape [T, B, n_acts]
    values: predicted v, in shape [T, B]
    rewards: rewards from actor, in shape [T, B]
    discounts: discounts from actor, in shape [T, B]
    clip_rho_threshold: rho_bar for values fitting
    clip_pg_rho_threshold: rho_bar for the advantage used in pg loss

  Returns:
    Vtrace policy gradient loss.
  """
  with tf.device("/cpu:0"):
    if mask is None:
      log_ratio = oldneglogp - neglogp
    else:
      log_ratio = (oldneglogp - neglogp) * mask
    adv = vtrace_from_importance_weights(
      log_ratio[:-1], discounts[:-1], rewards[:-1], values[:-1], values[-1],
      clip_rho_threshold=clip_rho_threshold,
      clip_pg_rho_threshold=clip_pg_rho_threshold
    ).pg_advantages
    if mask is None:
      loss = tf.reduce_mean(tf.stop_gradient(adv) * neglogp[:-1])
    else:
      loss = tf.reduce_mean(mask[:-1] * tf.stop_gradient(adv) * neglogp[:-1])
  return loss


def supervised_loss(label, predict):
  soft_logits = tf.nn.softmax(predict)
  soft_logits = tf.clip_by_value(soft_logits, 0.000001, 0.999999)
  loss = -tf.reduce_sum(label * tf.log(soft_logits), axis=1)
  return loss
