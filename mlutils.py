import torch

def train_net(epochs, network, dataloader, optimizer, log_interval=100, save=True):
  train_losses = []
  train_counter = []
  network.train()
  for batch_idx, (data, target) in enumerate(dataloader):
    data = data.to('cuda')
    target = target.to('cuda')
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epochs, batch_idx * len(data), len(dataloader.dataset),
        100. * batch_idx / len(dataloader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epochs-1)*len(dataloader.dataset)))
    if save:
        torch.save(network.state_dict(), f'./results/model{batch_idx}.pth')
        torch.save(optimizer.state_dict(), f'./results/optimizer{batch_idx}.pth')


def test_net():
    pass