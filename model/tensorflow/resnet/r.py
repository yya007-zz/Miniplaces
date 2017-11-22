import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torch.autograd import Variable

from data_utils import MiniPlaces

batch_size = 100
load_size = 256
fine_size = 224
c = 3
data_path = '../../../data/'
# data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.00001
training_iters = 40000
num_epochs = 30
display = 20
best_save = 250
checkpoint = 20


def validation(model, loader):
  print('Checking validation accuracy...')
  model.eval()

  total = 0
  top1_correct = 0
  top5_correct = 0

  for img_batch, label_batch in loader:
    img_batch = Variable(img_batch.cuda())
    output = model(img_batch)

    _, predict = torch.max(output.data, 1)
    top1_correct += torch.sum(predict.cpu() == label_batch)

    predict5 = output.data.cpu().topk(5, 1)[1]
    top5_correct += predict5.eq(label_batch.unsqueeze(1).expand_as(predict5)).sum()

    total += label_batch.size(0)

  acc1 = top1_correct / total
  acc5 = top5_correct / total
  print('top 1 accuracy: %.6f' % acc1)
  print('top 5 accuracy: %.6f' % acc5)
  model.train()
  return acc1, acc5



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

train_set = MiniPlaces(data_path, 'train', transform)
val_set = MiniPlaces(data_path, 'val', transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

model = resnet50(num_classes=100)
model.avgpool = torch.nn.AdaptiveAvgPool2d(1)

step = 0
stats = {'iter': [], 'loss': [], 'checkpoint': [], 'acc1': [], 'acc5':[]}

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

model = model.cuda()
criterion = criterion.cuda()

model.train()
best = 0.7
for epoch in range(num_epochs):
  for img_batch, label_batch in train_loader:

    img_batch = Variable(img_batch.cuda())
    label_batch = Variable(label_batch.cuda())

    logits = model(img_batch)
    loss = criterion(logits, label_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    step += 1

    stats['iter'].append(step)
    stats['loss'].append(loss.data[0])

    if step % display == 0:
      print('Running epoch %d. iteration %d. loss = %.4f.' % (epoch+1, step, loss.data[0]))
    
    if step % best_save == 0:
      acc1, acc5 = validation(model, val_loader)
      if acc5 > best:
        print('saving best model')
        best = acc5
        best_model = {
          'stats': stats,
          'model': model.state_dict()
        }
        torch.save(best_model, 'best_model.pt')
        print('Saving best model')

    elif step % checkpoint == 0 :
      acc1, acc5 = validation(model, val_loader)
      stats['checkpoint'].append(step)
      stats['acc1'].append(acc1)
      stats['accs'].append(acc5)

      checkpoint = {
        'stats': stats,
        'model_state': model.state_dict()
      }
      torch.save(checkpoint, 'checkpoint.pt')
      
result = {
  'stats': stats,
  'model_state': model.state_dict()
}
print('Finished training.')
torch.save(result, 'result.pt')


