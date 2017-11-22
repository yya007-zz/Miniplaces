import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torch.autograd import Variable
import time
import argparse

from data_utils import MiniPlaces

def to_var(x):
  if torch.cuda.is_available():
    x = x.cuda()
  return Variable(x)

def top_k_correct(preds, labels, k):
  '''
  preds: N by K Tensor
  labels: N by 1 LongTensor
  '''
  topk_idx = preds.topk(k, 1)[1]
  return topk_idx.eq(labels.unsqueeze(1).expand_as(topk_idx)).sum()

def main(args):
  data_path = args.data_path

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
  train_set = MiniPlaces(data_path, 'train', transform)
  train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True)
  val_set = MiniPlaces(data_path, 'val', transform)
  val_loader = DataLoader(val_set, batch_size = args.batch_size)

  if args.checkpoint != None:
    checkpoint = torch.load(args.checkpoint)
    model_type = checkpoint['args']['model']
    print(model_type)
  else:
    model_type = args.model

  if model_type == 'resnet34':
    model = resnet34(num_classes=100)
  elif model_type == 'resnet50':
    model = resnet50(num_classes=100)
  elif model_type == 'resnet101':
    model = resnet101(num_classes=100)
  else:
    model_type = 'resnet18'
    model = resnet18(num_classes=100)
  model.avgpool = torch.nn.AdaptiveAvgPool2d(1)

  if args.checkpoint != None:
    model.load_state_dict(checkpoint['model_state'])
    count = checkpoint['stats']['iter'][-1]
    stats = checkpoint['stats']
  else:
    count = 0
    stats = {'iter': [], 'loss': [], 'time': [], 'checkpoints':[], 'top1_accs': [], 'top5_accs':[]}

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

  if torch.cuda.is_available():
    print('training %s with %d GPUs' % (model_type, torch.cuda.device_count()))
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    criterion = criterion.cuda()
  else:
    print('training %s with CPU' % model_type)

  model.train()
  clock = time.time()
  count_0 = count
  best_top5_acc = 0
  for epoch in range(args.num_epochs):
    for img_batch, label_batch in train_loader:

      img_batch = to_var(img_batch)
      label_batch = to_var(label_batch)

      logits = model(img_batch)
      loss = criterion(logits, label_batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      count += 1
      time_ellapsed = time.time() - clock

      stats['iter'].append(count)
      stats['loss'].append(loss.data[0])
      stats['time'].append(time_ellapsed)

      if count % args.show_every == 0:
        print('running epoch %d. total iteration %d. loss = %.4f. estimated time remaining = %ds'
         % (epoch+1, count, loss.data[0], int(time_ellapsed * (len(train_set) / args.batch_size * args.num_epochs - count + count_0) / (count - count_0))))
      if count % args.save_every == 0:
        top1_acc, top5_acc = check_accuracy(model, val_loader)
        stats['checkpoints'].append(count)
        stats['top1_accs'].append(top1_acc)
        stats['top5_accs'].append(top5_acc)
        if top5_acc > best_top5_acc:
          print('saving best_model')
          best_top5_acc = top5_acc
          best_model = {
            'args': args.__dict__,
            'stats': stats,
            'model_state': model.state_dict()
          }
          torch.save(best_model, args.result_path + 'best_model.pt')
        print('saving checkpoint')
        checkpoint = {
          'args': args.__dict__,
          'stats': stats,
          'model_state': model.state_dict()
        }
        torch.save(checkpoint, args.result_path + 'checkpoint.pt')
        
  result = {
    'args': args.__dict__,
    'stats': stats,
    'model_state': model.state_dict()
  }
  print('finished training. total iteration = %d. saving result to %s'
    % (count, args.result_path + 'result.pt'))
  torch.save(result, args.result_path + 'result.pt')

def check_accuracy(model, loader):
  print('checking validation accuracy...')
  model.eval()

  total = 0
  correct_top1 = 0
  correct_top5 = 0
  count = 0
  for img_batch, label_batch in loader:
    img_batch = to_var(img_batch)
    output = model(img_batch)
    _, pred = torch.max(output.data, 1)
    correct_top1 += torch.sum(pred.cpu() == label_batch)
    correct_top5 += top_k_correct(output.data.cpu(), label_batch, 5)
    total += label_batch.size(0)
    count += 1
    print(count, '...')

  top1_acc = correct_top1 / total
  top5_acc = correct_top5 / total
  print('top 1 accuracy: %.6f' % top1_acc)
  print('top 5 accuracy: %.6f' % top5_acc)
  model.train()
  return top1_acc, top5_acc

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', type=str, default=None,
                        help='start from previous checkpoint')
  parser.add_argument('--model', type=str, default='resnet34',
                        help='network architecture')
  parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size for training')
  parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate for adam optimizer')
  parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of training epochs')
  parser.add_argument('--save_every', type=int, default=500,
                        help='save model every ? iteration')
  parser.add_argument('--show_every', type=int, default=20,
                        help='display log info every ? iteration')
  parser.add_argument('--data_path', type=str, default='data/',
                        help='data directory')
  parser.add_argument('--result_path', type=str, default='results/',
                        help='save result and checkpoint to')
  args = parser.parse_args()
  main(args)