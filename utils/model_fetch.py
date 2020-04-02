from models import resnet_model

def fetch_teacher(model_name):
    if model_name == 'resnet34':
        return resnet_model.ResNet34()
    elif model_name == 'resnet50':
        return resnet_model.ResNet50()
    elif model_name == 'resnet101':
        return resnet_model.ResNet101()
    elif model_name == 'resnet152':
        return resnet_model.ResNet152()
    else:
        raise ValueError("Not a valid model name")

def fetch_student(model_name):
    if model_name == 'resnet18':
        return resnet_model.ResNet18()
    elif model_name == 'resnet50':
        return resnet_model.ResNet50()
    elif model_name == 'resnet101':
        return resnet_model.ResNet101()
    elif model_name == 'resnet34':
        return resnet_model.ResNet34()
    else:
        raise ValueError("Not a valid model name")
