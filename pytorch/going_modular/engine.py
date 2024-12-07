from numpy import argmax
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device:torch.device) -> Tuple[float, float]:
    print("train_step , device = ", device)
    model.to(device)
    model.train()
    # train_loss,train_acc = 0.0, 0.0
    train_loss, train_correct = 0.0, 0
    total = 0
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X.size(0)  # 累积损失，乘以批量大小
        preds = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_correct += (preds == y).sum().item()  # 累积正确预测数量
        total += y.size(0)  # 累积总样本数量
    
    train_loss /= total  # 平均损失
    train_acc = train_correct / total  # 准确率
    #     train_loss += loss.item()
    #     train_acc += (torch.argmax(torch.softmax(y_pred,dim=1),dim=1) == y).sum().item() / len(y_pred)
    
    # train_loss /= len(dataloader.dataset)
    # train_acc /= len(dataloader.dataset)
    return train_loss, train_acc

def test_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              device:torch.device) -> Tuple[float, float]:
    model.to(device)
    model.eval()
    
    test_loss, test_correct = 0.0, 0.0
    total = 0
    with torch.inference_mode():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            
            test_loss += loss.item() * X.size(0)  # 累积损失，乘以批量大小
            preds = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_correct += (preds == y).sum().item()  # 累积正确预测数量
            total += y.size(0)  # 累积总样本数量
        
    test_loss /= total  # 平均损失
    test_acc = test_correct / total  # 准确率
    #         test_loss += loss.item()
    #         test_acc += (torch.argmax(torch.softmax(y_pred,dim=1),dim=1) == y).sum().item() / len(y_pred)
        
    # test_loss /= len(dataloader.dataset)
    # test_acc /= len(dataloader.dataset)
    return test_loss, test_acc

def train(model:torch.nn.Module,
          train_dataloader:torch.utils.data.DataLoader,
          test_dataloader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          loss_fn:torch.nn.Module,
          epochs:int,
          device:torch.device) ->Dict[str,list]:
    results = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        print(
            f"Epoch:{epoch+1} |"
            f"train_loss:{train_loss:.4f} |"
            f"train_acc:{train_acc:.2f} | "
            f"test_loss:{test_loss:.4f} | "
            f"test_acc:{test_acc:.2f}"
        )
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        
    return results

