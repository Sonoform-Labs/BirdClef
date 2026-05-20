import math
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class Cfg:
    prototypes_per_class: int = 5
    num_classes: int = 234
    channels: int = 1024
    height: int = 1
    width: int = 1
    topk_k: int = 1
    margin: float = None
    add_on_layers_type: str = "identity"
    incorrect_class_connection: float = None
    correct_class_connection: float = 1.0
    bias_last_layer: float = -2.0
    non_negative_last_layer: bool = True
    embedded_spectrogram_height: int = None
    input_vector_length: float = 64.0
    n_eps_channels: int = 2
    epsilon_val: float = 1e-4
    relu_on_cos: bool = True
    num_prototypes_after_pruning: int = None
    prototype_class_identity = None
    @property
    def use_bias_last_layer(self):
        return bool(self.bias_last_layer)


class AsymmetricLossMultiLabel(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=False, reduction="mean"):
        super().__init__()
        self.gamma_neg = gamma_neg; self.gamma_pos = gamma_pos; self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps; self.reduction = reduction

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x); xs_pos = x_sigmoid; xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y; pt1 = xs_neg * (1 - y); pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w
        if self.reduction == "mean":
            return -loss.mean()
        if self.reduction == "sum":
            return -loss.sum()
        return -loss


class NonNegativeLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        return nn.functional.linear(input, torch.relu(self.weight), self.bias)


class LinearLayerWithoutNegativeConnections(nn.Module):
    __constants__ = ["in_features", "out_features", "bias"]

    def __init__(self, in_features, out_features, bias=True, non_negative=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.non_negative = non_negative
        self.features_per_output_class = in_features // out_features
        assert in_features % out_features == 0, f"{in_features=} must be divisible by {out_features=}"
        self.weight = nn.Parameter(torch.empty((out_features, self.features_per_output_class), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        batch_size = input.size(0)
        reshaped_input = input.view(batch_size, self.out_features, self.features_per_output_class)
        weight = torch.relu(self.weight) if self.non_negative else self.weight
        output = torch.einsum("bof,of->bo", reshaped_input, weight)
        if self.bias is not None:
            output += self.bias
        return output


class AudioProtoNetClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prototypes_per_class = config.prototypes_per_class
        self.num_classes = config.num_classes
        self.num_prototypes = self.prototypes_per_class * self.num_classes
        self.num_prototypes_after_pruning = config.num_prototypes_after_pruning
        self.margin = config.margin
        self.relu_on_cos = config.relu_on_cos
        self.incorrect_class_connection = config.incorrect_class_connection
        self.correct_class_connection = config.correct_class_connection
        self.input_vector_length = config.input_vector_length
        self.n_eps_channels = config.n_eps_channels
        self.epsilon_val = config.epsilon_val
        self.topk_k = config.topk_k
        self.bias_last_layer = config.bias_last_layer
        self.non_negative_last_layer = config.non_negative_last_layer
        self.embedded_spectrogram_height = config.embedded_spectrogram_height
        self.use_bias_last_layer = config.use_bias_last_layer
        self.prototype_class_identity = torch.arange(self.num_prototypes) // self.prototypes_per_class
        self.prototype_shape = (self.num_prototypes, config.channels, config.height, config.width)
        self._setup_add_on_layers(add_on_layers_type=config.add_on_layers_type)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.frequency_weights = None
        if self.embedded_spectrogram_height is not None:
            self.frequency_weights = nn.Parameter(
                torch.full((self.num_prototypes, self.embedded_spectrogram_height), 3.0))
        if self.incorrect_class_connection:
            if self.non_negative_last_layer:
                self.last_layer = NonNegativeLinear(self.num_prototypes, self.num_classes, bias=self.use_bias_last_layer)
            else:
                self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=self.use_bias_last_layer)
        else:
            self.last_layer = LinearLayerWithoutNegativeConnections(
                in_features=self.num_prototypes, out_features=self.num_classes,
                non_negative=self.non_negative_last_layer)

    def _setup_add_on_layers(self, add_on_layers_type):
        if add_on_layers_type == "identity":
            self.add_on_layers = nn.Sequential(nn.Identity())
        elif add_on_layers_type == "upsample":
            self.add_on_layers = nn.Upsample(scale_factor=2, mode="bilinear")
        else:
            raise NotImplementedError(add_on_layers_type)

    def cos_activation(self, x, prototypes_of_wrong_class=None):
        input_vector_length = self.input_vector_length
        normalizing_factor = (self.prototype_shape[-2] * self.prototype_shape[-1]) ** 0.5
        epsilon_channel_x = torch.full((x.shape[0], self.n_eps_channels, x.shape[2], x.shape[3]),
                                       self.epsilon_val, device=x.device, requires_grad=False)
        x = torch.cat((x, epsilon_channel_x), dim=-3)
        x_length = torch.sqrt(torch.sum(x ** 2, dim=-3, keepdim=True) + self.epsilon_val)
        x_normalized = (input_vector_length * x / x_length) / normalizing_factor
        epsilon_channel_p = torch.full((self.prototype_shape[0], self.n_eps_channels,
                                        self.prototype_shape[2], self.prototype_shape[3]),
                                       self.epsilon_val, device=self.prototype_vectors.device, requires_grad=False)
        appended_protos = torch.cat((self.prototype_vectors, epsilon_channel_p), dim=-3)
        prototype_vector_length = torch.sqrt(torch.sum(appended_protos ** 2, dim=-3, keepdim=True) + self.epsilon_val)
        normalized_prototypes = appended_protos / (prototype_vector_length + self.epsilon_val)
        normalized_prototypes /= normalizing_factor
        activations_dot = nn.functional.conv2d(x_normalized, normalized_prototypes)
        marginless_activations = activations_dot / (input_vector_length * 1.01)
        if self.frequency_weights is not None:
            freq_weights = torch.sigmoid(self.frequency_weights)
            marginless_activations = marginless_activations * freq_weights[:, :, None]
        if self.margin is None or not self.training or prototypes_of_wrong_class is None:
            activations = marginless_activations
        else:
            wrong_class_margin = (prototypes_of_wrong_class * self.margin).view(
                x.size(0), self.prototype_vectors.size(0), 1, 1)
            wrong_class_margin = wrong_class_margin.expand(-1, -1, activations_dot.size(-2), activations_dot.size(-1))
            penalized_angles = torch.acos(activations_dot / (input_vector_length * 1.01)) - wrong_class_margin
            activations = torch.cos(torch.relu(penalized_angles))
        if self.relu_on_cos:
            activations = torch.relu(activations)
            marginless_activations = torch.relu(marginless_activations)
        return activations, marginless_activations

    def prototype_activations(self, x, prototypes_of_wrong_class=None):
        activations, marginless_activations = self.cos_activation(x, prototypes_of_wrong_class=prototypes_of_wrong_class)
        return activations, [marginless_activations, x]

    def forward(self, features, prototypes_of_wrong_class=None):
        features = self.add_on_layers(features)
        activations, additional_returns = self.prototype_activations(
            features, prototypes_of_wrong_class=prototypes_of_wrong_class)
        marginless_activations = additional_returns[0]
        topk_k = 1
        activations = activations.view(activations.shape[0], activations.shape[1], -1)
        topk_activations, _ = torch.topk(activations, topk_k, dim=-1)
        mean_activations = torch.mean(topk_activations, dim=-1)
        marginless_max_activations = nn.functional.max_pool2d(
            marginless_activations,
            kernel_size=(marginless_activations.size()[2], marginless_activations.size()[3]))
        marginless_max_activations = marginless_max_activations.view(-1, self.num_prototypes)
        logits = self.last_layer(mean_activations)
        marginless_logits = self.last_layer(marginless_max_activations)
        return logits, [mean_activations, marginless_logits, None, marginless_max_activations, marginless_activations]

    def get_prototype_orthogonalities(self):
        pv = self.prototype_vectors.view(self.num_prototypes, -1)
        plen = torch.sqrt(torch.sum(pv ** 2, dim=1, keepdim=True) + self.epsilon_val)
        normp = pv / (plen + self.epsilon_val)
        normp = normp.view(self.num_classes, self.prototypes_per_class,
                           self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3])
        orth = torch.matmul(normp, normp.transpose(1, 2))
        eye = torch.eye(normp.shape[1], device=orth.device).unsqueeze(0).repeat(self.num_classes, 1, 1)
        return orth - eye

    def set_last_layer_incorrect_connection(self, incorrect_strength=None):
        if incorrect_strength is None:
            if isinstance(self.last_layer, LinearLayerWithoutNegativeConnections):
                self.last_layer.weight.data.fill_(self.correct_class_connection)
            else:
                raise ValueError("last_layer is not LinearLayerWithoutNegativeConnections")
        else:
            pos = torch.zeros(self.num_classes, self.num_prototypes)
            pos[self.prototype_class_identity, torch.arange(self.num_prototypes)] = 1
            neg = 1 - pos
            self.last_layer.weight.data.copy_(self.correct_class_connection * pos + incorrect_strength * neg)
        if self.last_layer.bias is not None:
            self.last_layer.bias.data.fill_(self.bias_last_layer)


if __name__ == "__main__":
    cfg = Cfg(num_classes=234, prototypes_per_class=5, channels=1024)
    head = AudioProtoNetClassificationHead(cfg)
    head.set_last_layer_incorrect_connection(incorrect_strength=None)
    x = torch.randn(2, 1024, 4, 15)
    logits, info = head(x)
    npar = sum(p.numel() for p in head.parameters())
    print(f"OK head: logits {tuple(logits.shape)} (expect [2,234])  protos {tuple(head.prototype_vectors.shape)} "
          f"last_layer {type(head.last_layer).__name__}  params {npar/1e6:.2f}M")
    print(f"  prototype_class_identity[:7]={head.prototype_class_identity[:7].tolist()} (class-specific: 0,0,0,0,0,1,1)")
