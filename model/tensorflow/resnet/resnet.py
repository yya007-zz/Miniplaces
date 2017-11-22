# import torch
from torchvision.models.resnet import resnet50
from torch.autograd import Variable
from DataLoader import *
from DataLoaderNoise import DataLoaderDiskRandomize
import time
 
# batch_size = 100
# load_size = 256
# fine_size = 224
# c = 3
# data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# # Training Parameters
# learning_rate = 0.0001
# training_iters = 40000
# show_every = 50

# opt_data_train = {
#   #'data_h5': 'miniplaces_256_train.h5',
#   'data_root': '../../../data/images/',   # MODIFY PATH ACCORDINGLY
#   'data_list': '../../../data/train.txt', # MODIFY PATH ACCORDINGLY
#   'load_size': load_size,
#   'fine_size': fine_size,
#   'data_mean': data_mean,
#   'randomize': True,
#   'perm' : True
#   }

# opt_data_val = {
#   #'data_h5': 'miniplaces_256_val.h5',
#   'data_root': '../../../data/images/',   # MODIFY PATH ACCORDINGLY
#   'data_list': '../../../data/val.txt',   # MODIFY PATH ACCORDINGLY
#   'load_size': load_size,
#   'fine_size': fine_size,
#   'data_mean': data_mean,
#   'randomize': False,
#   'perm' : False
#   }

# opt_data_test = {
#   #'data_h5': 'miniplaces_256_val.h5',
#   'data_root': '../../../data/images/',   # MODIFY PATH ACCORDINGLY
#   'data_list': '../../../data/test.txt',   # MODIFY PATH ACCORDINGLY
#   'load_size': load_size,
#   'fine_size': fine_size,
#   'data_mean': data_mean,
#   'randomize': False,
#   'perm' : False
#   }

# loader_train = DataLoaderDiskRandomize(**opt_data_train)
# loader_val = DataLoaderDisk(**opt_data_val)
# loader_test = DataLoaderDisk(**opt_data_test)

# def to_var(x):
#   if torch.cuda.is_available():
#     x = x.cuda()
#   return Variable(x)

# def top_k_correct(preds, labels, k):
#   '''
#   preds: N by K Tensor
#   labels: N by 1 LongTensor
#   '''
#   topk_idx = preds.topk(k, 1)[1]
#   return topk_idx.eq(labels.unsqueeze(1).expand_as(topk_idx)).sum()

# def main():
# # Construct dataloader
#   model = resnet50(num_classes=100)

#   count = 0
#   stats = {'iter': [], 'loss': [], 'time': [], 'checkpoints':[], 'top1_accs': [], 'top5_accs':[]}

#   criterion = torch.nn.CrossEntropyLoss()
#   optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

#   model = model.cuda()
#   model = torch.nn.DataParallel(model)
#   criterion = criterion.cuda()

#   model.train()
#   clock = time.time()

#   step = 0
#   while step < training_iters:
#       # Load a batch of training data
#       img_batch, label_batch = loader_train.next_batch(batch_size)

#       img_batch = to_var(img_batch)
#       label_batch = to_var(label_batch)

#       logits = model(img_batch)
#       loss = criterion(logits, label_batch)

#       optimizer.zero_grad()
#       loss.backward()
#       optimizer.step()

#       step += 1
#       time_ellapsed = time.time() - clock

#       stats['iter'].append(step)
#       stats['loss'].append(loss.data[0])
#       stats['time'].append(time_ellapsed)

#       if step % show_every == 0:
#         print('running iteration %d. loss = %.4f.' % (step, loss.data[0]))
#       if step % args.save_every == 0:
#         top1_acc, top5_acc = check_accuracy(model, loader_val, 10000)
#         stats['checkpoints'].append(step)
#         stats['top1_accs'].append(top1_acc)
#         stats['top5_accs'].append(top5_acc)
#         if top5_acc > best_top5_acc:
#           print('saving best_model')
#           best_top5_acc = top5_acc
#           best_model = {
#             'stats': stats,
#             'model_state': model.state_dict()
#           }
#           torch.save(best_model, 'best_model.pt')
#         print('saving checkpoint')
#         checkpoint = {
#           'stats': stats,
#           'model_state': model.state_dict()
#         }
#         torch.save(checkpoint, 'checkpoint.pt')
        
#   result = {
#     'stats': stats,
#     'model_state': model.state_dict()
#   }
#   print('finished training. total iteration = %d. saving result to %s'
#     % (step, 'result.pt'))
#   torch.save(result, 'result.pt')

# def check_accuracy(model, loader, batch_size):
#   print('checking validation accuracy...')
#   model.eval()

#   num_batch = loader.size()//batch_size+1
#   loader.reset()
#   result=[]
#   for i in range(num_batch):
#     img_batch, label_batch = loader.next_batch(batch_size)
#     output = model(img_batch)
#     _, pred = torch.max(output.data, 1)
#     correct_top1 += torch.sum(pred.cpu() == label_batch)
#     correct_top5 += top_k_correct(output.data.cpu(), label_batch, 5)
#     total += label_batch.size(0)
#     count += 1
#     print(count, '...')

#   top1_acc = correct_top1 / total
#   top5_acc = correct_top5 / total
#   print('top 1 accuracy: %.6f' % top1_acc)
#   print('top 5 accuracy: %.6f' % top5_acc)
#   model.train()
#   return top1_acc, top5_acc

def main_sudo():
  print("worked")

if __name__ == '__main__':
  main_sudo()