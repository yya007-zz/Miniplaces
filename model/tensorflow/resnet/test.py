import argparse
import torch
from torch.autograd import Variable
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torch.utils.data import DataLoader
from torchvision import transforms
from data_utils import Miniplaces

def to_var(x):
  if torch.cuda.is_available():
    x = x.cuda()
  return Variable(x)

def main(args):
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
  miniplaces = Miniplaces(args.data_path, 'test', transform)
  places_loader = DataLoader(miniplaces, batch_size = 1, shuffle=False)
  
  checkpoint = torch.load(args.model_path)

  if checkpoint['args']['model'] == 'resnet34':
    model = resnet34(num_classes=100)
  elif checkpoint['args']['model'] == 'resnet50':
    model = resnet50(num_classes=100)
  elif checkpoint['args']['model'] == 'resnet101':
    model = resnet101(num_classes=100)
  else:
    model = resnet18(num_classes=100)
  model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
  if torch.cuda.is_available():
    model = model.cuda()
    model = torch.nn.DataParallel(model)
  model.load_state_dict(checkpoint['model_state'])
  model.eval()

  model_dir = '/'.join(args.model_path.split('/')[:-1]) + '/'
  with open('%stest_pred.txt' % model_dir, 'w') as f:
    for i, (img, _) in enumerate(places_loader):
      img = to_var(img)
      output = model(img)
      pred = output.topk(5)[1].data
      filename = 'test/' + miniplaces.imgs[i].split('/')[-1]
      print('%s %d %d %d %d %d' % (filename, pred[0, 0], pred[0, 1], pred[0, 2], pred[0, 3], pred[0, 4]))
      f.write('%s %d %d %d %d %d\n' % (filename, pred[0, 0], pred[0, 1], pred[0, 2], pred[0, 3], pred[0, 4]))