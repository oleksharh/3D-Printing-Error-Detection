from sklearn.metrics import confusion_matrix
import torch

preds = torch.load("test/preds_02-09_12-03-26.pt").cpu().numpy()
targets = torch.load("test/targets_02-09_12-03-26.pt").cpu().numpy()


cm = confusion_matrix(targets[0], preds[0])
print(cm)