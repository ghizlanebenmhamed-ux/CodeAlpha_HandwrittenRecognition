import json,random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from model import DigitCNN,ROOT

def main():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    data=np.load(ROOT/'data'/'mnist.npz')
    x=data['x_train'];y=data['y_train']
    train_ids,val_ids=train_test_split(np.arange(len(x)),test_size=5000,stratify=y,random_state=42)
    def loader(ids,shuffle=False):
        return DataLoader(TensorDataset(torch.from_numpy(x[ids].astype(np.float32)/255).unsqueeze(1),torch.from_numpy(y[ids].astype(np.int64))),batch_size=128,shuffle=shuffle)
    model=DigitCNN();optimizer=torch.optim.Adam(model.parameters(),lr=.001);loss_fn=nn.CrossEntropyLoss()
    history=[];best=-1
    for epoch in range(1,5):
        model.train();total=0
        for images,labels in loader(train_ids,True):
            optimizer.zero_grad();loss=loss_fn(model(images),labels);loss.backward();optimizer.step();total+=float(loss.detach())*len(labels)
        model.eval();correct=0
        with torch.no_grad():
            for images,labels in loader(val_ids):correct+=int((model(images).argmax(1)==labels).sum())
        accuracy=correct/len(val_ids);history.append(dict(epoch=epoch,training_loss=total/len(train_ids),validation_accuracy=accuracy))
        if accuracy>best:best=accuracy;torch.save(model.state_dict(),ROOT/'digit_cnn.pt')
        print(f'Epoch {epoch}: validation accuracy {accuracy:.4f}',flush=True)
    model.load_state_dict(torch.load(ROOT/'digit_cnn.pt',weights_only=True));model.eval()
    predicted=[]
    with torch.no_grad():
        for batch in np.array_split(data['x_test'],40):predicted.extend(model(torch.from_numpy(batch.astype(np.float32)/255).unsqueeze(1)).argmax(1).tolist())
    metrics=dict(training_samples=len(train_ids),validation_samples=len(val_ids),test_samples=len(data['y_test']),test_accuracy=float(accuracy_score(data['y_test'],predicted)),validation_best=best,epochs=4,seed=42,history=history,confusion_matrix=confusion_matrix(data['y_test'],predicted).tolist())
    (ROOT/'results'/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (ROOT/'results'/'evaluation.txt').write_text(classification_report(data['y_test'],predicted,digits=4))
    np.savez_compressed(ROOT/'data'/'test_samples.npz',images=data['x_test'][:200],labels=data['y_test'][:200])
    print('Test accuracy:',metrics['test_accuracy'],flush=True)
if __name__=='__main__':main()
