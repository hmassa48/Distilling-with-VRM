from models import resnet_model
from models import preact_resnet_model

_MODEL_DICT = {
    'resnet18': resnet_model.ResNet18(),
    'resnet34': resnet_model.ResNet34(),
    'resnet50': resnet_model.ResNet50(),
    'resnet101': resnet_model.ResNet101(),
    'resnet152': resnet_model.ResNet152(),
    'preact_resnet18': preact_resnet_model.PreActResNet18(),
    'preact_resnet34': preact_resnet_model.PreActResNet34(),
    'preact_resnet50': preact_resnet_model.PreActResNet50(),
    'preact_resnet101': preact_resnet_model.PreActResNet101(),
    'preact_resnet152': preact_resnet_model.PreActResNet152(),
}


def _invalid_model_name():
    raise ValueError("Not a valid model name")


def fetch_teacher(model_name, model_dicts=_MODEL_DICT):
    return model_dicts[model_name]


def fetch_student(model_name, model_dicts=_MODEL_DICT):
    return model_dicts[model_name]
