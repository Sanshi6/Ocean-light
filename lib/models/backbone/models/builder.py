import logging
import re
from collections.__init__ import OrderedDict
from copy import deepcopy
import torch.nn as nn
from lib.models.backbone.models.utils import _parse_ksize
from lib.models.backbone.models.units import *


# 将字符串变成对应模块的字典
def _decode_block_str(block_str):
    """ Decode block definition string
    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip_c4
    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.
    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
    #   ir: 倒残差
        ds：深度可分离卷积
        dsa: 深度可分离卷积 + 逐点卷积 + act
        cn： 卷积 + 批处理 + 激活函数

    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    n - activation fn ('re', 'r6', 'hs', or 'sw')
    Args:
        block_str: a string representation of block arguments.
    Returns:
        A list of block args (dicts)
    Raises:
        ValueError: if the string def not properly specified (TODO)
    """
    assert isinstance(block_str, str)
    ops = block_str.split('_')
    block_type = ops[0]  # take the block type off the front
    ops = ops[1:]
    options = {}
    noskip = False
    for op in ops:
        # string options being checked on individual basis, combine if they grow
        if op == 'noskip':
            noskip = True
        elif op.startswith('n'):
            # activation fn
            key = op[0]
            v = op[1:]
            if v == 're':
                value = nn.ReLU
            elif v == 'r6':
                value = nn.ReLU6
            elif v == 'sw':
                value = Swish
            else:
                continue
            options[key] = value
        else:
            # all numeric options
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    # if act_layer is None, the model default (passed to model init) will be used
    act_layer = options['n'] if 'n' in options else None
    exp_kernel_size = _parse_ksize(options['a']) if 'a' in options else 1
    pw_kernel_size = _parse_ksize(options['p']) if 'p' in options else 1
    fake_in_chs = int(options['fc']) if 'fc' in options else 0  # FIXME hack to deal with in_chs issue in TPU def

    num_repeat = int(options['r'])
    # each type of block has different valid arguments, fill accordingly
    if block_type == 'ir':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            exp_ratio=float(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_layer=act_layer,
            noskip=noskip,
        )
        if 'cc' in options:
            block_args['num_experts'] = int(options['cc'])
    elif block_type == 'ds' or block_type == 'dsa':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_layer=act_layer,
            pw_act=block_type == 'dsa',
            noskip=block_type == 'dsa' or noskip,
        )
    elif block_type == 'cn':
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options['k']),
            out_chs=int(options['c']),
            stride=int(options['s']),
            act_layer=act_layer,
        )
    else:
        assert False, 'Unknown block type (%s)' % block_type

    return block_args, num_repeat



# 更新字典参数
def modify_block_args(block_args, kernel_size, exp_ratio):
    # kernel_size: 3,5,7
    # exp_ratio: 4,6
    block_type = block_args['block_type']
    # each type of block has different valid arguments, fill accordingly
    if block_type == 'cn':
        block_args['kernel_size'] = kernel_size
    elif block_type == 'er':
        block_args['exp_kernel_size'] = kernel_size
    else:
        block_args['dw_kernel_size'] = kernel_size

    if block_type == 'ir' or block_type == 'er':
        block_args['exp_ratio'] = exp_ratio
    return block_args


# 这个函数控制网络的深度，这个要注意，因为网络每个stage的深度设置成多少，本身就是我做这个东西的目的
#  stack_args, 不使用
#  repeats: 本身要重复的列表 [2,3,5]
#  depth multiplier: 1.0
#  ceil
def _scale_stage_depth(stack_args, repeats, depth_multiplier=1.0, depth_trunc='ceil'):
    """ Per-stage depth scaling
    Scales the block repeats in each stage. This depth scaling impl maintains
    compatibility with the EfficientNet scaling method, while allowing sensible
    scaling for other models that may have multiple block arg definitions in each stage.
    """

    # We scale the total repeat count for each stage, there may be multiple
    # block arg defs per stage so we need to sum.
    num_repeat = sum(repeats)
    if depth_trunc == 'round':
        # Truncating to int by rounding allows stages with few repeats to remain
        # proportionally smaller for longer. This is a good choice when stage definitions
        # include single repeat stages that we'd prefer to keep that way as long as possible
        num_repeat_scaled = max(1, round(num_repeat * depth_multiplier))
    else:
        # The default for EfficientNet truncates repeats to int via 'ceil'.
        # Any multiplier > 1.0 will result in an increased depth for every stage.
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))

    # Proportionally distribute repeat count scaling to each block definition in the stage.
    # Allocation is done in reverse as it results in the first block being less likely to be scaled.
    # The first block makes less sense to repeat in most of the arch definitions.
    repeats_scaled = []
    for r in repeats[::-1]:             # 将 repeat 倒着放
        rs = max(1, round((r / num_repeat * num_repeat_scaled)))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    repeats_scaled = repeats_scaled[::-1]

    # Apply the calculated scaling to each block arg in the stage
    sa_scaled = []
    # 这里指重复的次数，stack_args 有可能是每个阶段block的配置，后面的scaled算出来它们的重复次数
    for ba, rep in zip(stack_args, repeats_scaled):
        sa_scaled.extend([deepcopy(ba) for _ in range(rep)])
    return sa_scaled


# block_strings 是一个列表，将这个列表变成 block config + re, 然后重复几次
def decode_arch_def(arch_def, depth_multiplier=1.0, depth_trunc='ceil', experts_multiplier=1):
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            ba, rep = _decode_block_str(block_str)
            if ba.get('num_experts', 0) > 0 and experts_multiplier > 1:
                ba['num_experts'] *= experts_multiplier
            stack_args.append(ba)
            repeats.append(rep)
        arch_args.append(_scale_stage_depth(stack_args, repeats, depth_multiplier, depth_trunc))
    return arch_args


# child net builder
class ChildNetBuilder:
    """ Build Trunk Blocks
    """
    # channel multiplier: 通道数乘以多少
    # channel divisor: 通道可以被8整除吗
    # channel min: 通道最小是多少
    # output stride: 输出层的stride是多少
    # pad type
    # act layer: 在哪一层加入激活函数
    # se： se字符串
    # drop_path_rate:这个要看一眼


    def __init__(self, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 output_stride=32, pad_type='', act_layer=None, se_kwargs=None,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0., feature_location='',
                 verbose=False):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.act_layer = act_layer
        self.se_kwargs = se_kwargs
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.drop_path_rate = drop_path_rate
        self.feature_location = feature_location
        assert feature_location in ('pre_pwl', 'post_exp', '')
        self.verbose = verbose

        # state updated during build, consumed by model
        self.in_chs = None
        self.features = OrderedDict()

    # 对齐通道 通道乘数， 对齐最小倍数， 通道最小数， chs是当前通道吧
    def _round_channels(self, chs):
        return round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    #
    def _make_block(self, ba, block_idx, block_count):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self._round_channels(ba['out_chs'])
        if 'fake_in_chs' in ba and ba['fake_in_chs']:
            # FIXME this is a hack to work around mismatch in origin impl input filters
            ba['fake_in_chs'] = self._round_channels(ba['fake_in_chs'])
        ba['norm_layer'] = self.norm_layer
        ba['norm_kwargs'] = self.norm_kwargs
        ba['pad_type'] = self.pad_type
        # block act fn overrides the model default
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'] is not None else self.act_layer
        assert ba['act_layer'] is not None
        if bt == 'ir':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            if self.verbose:
                logging.info('  InvertedResidual {}, Args: {}'.format(block_idx, str(ba)))
            block = InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            if self.verbose:
                logging.info('  DepthwiseSeparable {}, Args: {}'.format(block_idx, str(ba)))
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'cn':
            if self.verbose:
                logging.info('  ConvBnAct {}, Args: {}'.format(block_idx, str(ba)))
            block = ConvBnAct(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']  # update in_chs for arg of next block

        return block

    def __call__(self, in_chs, model_block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            model_block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
            model_block_args：这玩意儿是一个双重列表第一重是一个stage，第一个列表中的列表是第一个stage的配置
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        # 记录
        if self.verbose:
            logging.info('Building model trunk with %d stages...' % len(model_block_args))

        # 输入通道
        self.in_chs = in_chs
        # 总的块数量：
        total_block_count = sum([len(x) for x in model_block_args])
        # 总的块的 index
        total_block_idx = 0

        # 当前 stride
        current_stride = 2
        # 当前 dilation
        current_dilation = 1
        # 特征index
        feature_idx = 0
        stages = []

        # outer list of block_args defines the stacks ('stages' by some conventions)
        for stage_idx, stage_block_args in enumerate(model_block_args):
            # 是否是 last stack
            last_stack = stage_idx == (len(model_block_args) - 1)

            # 是否记录
            if self.verbose:
                logging.info('Stack: {}'.format(stage_idx))
            # 断言 stage block args 是一个 list
            assert isinstance(stage_block_args, list)


            blocks = []
            # 每一个 stage 都含有一个 list
            # each stack (stage) contains a list of block arguments
            for block_idx, block_args in enumerate(stage_block_args):
                # 判断是不是 last block
                last_block = block_idx == (len(stage_block_args) - 1)
                # no features extracted
                extract_features = ''  # No features extracted
                # 是否记录
                if self.verbose:
                    logging.info(' Block: {}'.format(block_idx))

                # Sort out stride, dilation, and feature extraction details
                assert block_args['stride'] in (1, 2)

                # block index， 第一个 block(conv) stride = 2，这里index是从0开始的
                if block_idx >= 1:
                    # only the first block in any stack can have a stride > 1
                    block_args['stride'] = 1

                # 是否 extract
                do_extract = False
                # 在哪里进行特征提取
                if self.feature_location == 'pre_pwl':
                    if last_block:      # 如果是最后一个block
                        next_stage_idx = stage_idx + 1  # 接下来的stage index 数字
                        if next_stage_idx >= len(model_block_args):  # 如果后面已经没有stage
                            do_extract = True           # 进行特征提取
                        else:
                            # 否则，只有当下一个stage的第一个block的stride>1才进行特征提取
                            do_extract = model_block_args[next_stage_idx][0]['stride'] > 1  #
                elif self.feature_location == 'post_exp':
                    # 只有当作当前 conv 的 block的stride>1且是最后一个stage的最后一个block的时候才进行特征提取
                    if block_args['stride'] > 1 or (last_stack and last_block):
                        do_extract = True

                # 不是很懂
                if do_extract:
                    extract_features = self.feature_location

                # current dilation
                next_dilation = current_dilation

                # block的stride > 1
                if block_args['stride'] > 1:
                    # 当前的 stride * 当前block的stride计算stride
                    next_output_stride = current_stride * block_args['stride']
                    # 如果stride已经超出了output限制的stride
                    if next_output_stride > self.output_stride:
                        # 将 next_dilation 变大， stride调整为1
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                        # 将 stride 转变成 dilation，表示在不扩大步长的情况下，增大感受野
                        if self.verbose:
                            logging.info('  Converting stride to dilation to maintain output_stride=={}'.format(
                                self.output_stride))
                    else:
                        # 没有超出，那么正常计算
                        current_stride = next_output_stride

                # dilation
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation

                # make clock xingxing
                # create the block
                # block_args, total_clock_idx, total block count
                block = self._make_block(block_args, total_block_idx, total_block_count)
                blocks.append(block)

                # stash feature module name and channel info for model feature extraction
                if extract_features:
                    feature_module = block.feature_module(extract_features)
                    if feature_module:
                        feature_module = 'blocks.{}.{}.'.format(stage_idx, block_idx) + feature_module
                    feature_channels = block.feature_channels(extract_features)
                    self.features[feature_idx] = dict(
                        name=feature_module,
                        num_chs=feature_channels
                    )
                    feature_idx += 1

                total_block_idx += 1  # incr global block idx (across all stacks)
            stages.append(nn.Sequential(*blocks))
        return stages


def _init_weight_goog(m, n='', fix_group_fanout=True, last_bn=None):
    """ Weight initialization as per Tensorflow official implementations.
    Args:
        m (nn.Module): module to init
        n (str): module name
        fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/ group convs
    Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
    * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    if isinstance(m, CondConv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        init_weight_fn = get_condconv_initializer(
            lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        init_weight_fn(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if n in last_bn:
            m.weight.data.zero_()
            m.bias.data.zero_()
        else:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def efficientnet_init_weights(model: nn.Module, init_fn=None, zero_gamma=False):
    last_bn = []
    if zero_gamma:
        prev_n = ''
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if ''.join(prev_n.split('.')[:-1]) != ''.join(n.split('.')[:-1]):
                    last_bn.append(prev_n)
                prev_n = n
        last_bn.append(prev_n)

    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n, last_bn=last_bn)


if __name__ == '__main__':
    # _decode_block_str("ir_r2_k3_s2_e1_i32_o16_se0.25_noskip_c4")
    # _scale_stage_depth([2, 2, 2], [2, 3, 4], 1.5, "ceil")
    ChildNetBuilder(channel_multiplier=1.0, channel_divisor=8, output_stride=16)
    print(1)
