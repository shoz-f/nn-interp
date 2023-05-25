import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

r18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
r18_scripted = torch.jit.script(r18)

dummy_input = torch.rand(1, 3, 224, 224)
unscripted_output = r18(dummy_input)
scripted_output = r18_scripted(dummy_input)

unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices

print('Python model top5 results:\n {}'.format(unscripted_top5))
print('TorchScript model top 5 results:\n {}'.format(scripted_top5))

r18_scripted.save('r18_scripted.pt')
