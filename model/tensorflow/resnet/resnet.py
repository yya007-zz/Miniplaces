import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.resnet import resnet18, resnet34, resnet50
from torch.autograd import Variable
import time

from processing import Miniplaces

batch_size = 256
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.0001
training_iters = 40000

opt_data_train = {
      #'data_h5': 'miniplaces_256_train.h5',
      'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
      'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
      'load_size': load_size,
      'fine_size': fine_size,
      'data_mean': data_mean,
      'randomize': True,
      'perm' : True
      }

  opt_data_val = {
      #'data_h5': 'miniplaces_256_val.h5',
      'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
      'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
      'load_size': load_size,
      'fine_size': fine_size,
      'data_mean': data_mean,
      'randomize': False,
      'perm' : False
      }

  opt_data_test = {
      #'data_h5': 'miniplaces_256_val.h5',
      'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
      'data_list': '../../data/test.txt',   # MODIFY PATH ACCORDINGLY
      'load_size': load_size,
      'fine_size': fine_size,
      'data_mean': data_mean,
      'randomize': False,
      'perm' : False
      }

loader_train = DataLoaderDiskRandomize(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
loader_test = DataLoaderDisk(**opt_data_test)

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
# Construct dataloader
  model = resnet50(num_classes=100)

  count = 0
  stats = {'iter': [], 'loss': [], 'time': [], 'checkpoints':[], 'top1_accs': [], 'top5_accs':[]}

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

  model = model.cuda()
  model = torch.nn.DataParallel(model)
  criterion = criterion.cuda()

  model.train()
  # clock = time.time()
  count_0 = count
  best_top5_acc = 0
  for epoch in range(num_epochs):
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