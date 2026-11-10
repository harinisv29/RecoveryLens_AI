import argparse
import copy
import json
import random
import sys
import time
import hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import ROOT,SEED
from models.temporal_transformer import TemporalTransformer
from models.gru_baseline import GRUClassifier
from training.evaluate import metrics,tune_threshold,bootstrap_intervals,save_error_analysis
from utils.experiments import run_paths

def collate(batch):
    xs,ys=zip(*batch)
    lengths=torch.tensor([len(x) for x in xs])
    padded=torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in xs],batch_first=True)
    return padded,torch.arange(padded.shape[1])[None,:]<lengths[:,None],torch.tensor(ys,dtype=torch.float32)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run'); parser.add_argument('--model',choices=['transformer','gru'],default='transformer'); parser.add_argument('--epochs',type=int,default=40); parser.add_argument('--batch-size',type=int,default=16); parser.add_argument('--lr',type=float,default=1e-4); parser.add_argument('--d-model',type=int,default=64); parser.add_argument('--heads',type=int,default=4); parser.add_argument('--layers',type=int,default=2); parser.add_argument('--dropout',type=float,default=.1); args=parser.parse_args()
    # Parse experiment options separately to retain backwards compatible CLI arguments.
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    paths=run_paths(args.run)
    checkpoint_path=paths['checkpoints']/f'{args.model}.pt'
    if args.run and checkpoint_path.exists(): raise RuntimeError('Named checkpoint exists; use a new run name')
    data=torch.load(paths['processed']/'dataset.pt',weights_only=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    loaders={k:DataLoader([(data['x'][i],data['y'][i]) for i in ids],batch_size=args.batch_size,shuffle=k=='train',collate_fn=collate) for k,ids in data['splits'].items()}
    architecture={'d_model':args.d_model,'heads':args.heads,'layers':args.layers,'dropout':args.dropout} if args.model=='transformer' else {}
    model=(TemporalTransformer(**architecture) if args.model=='transformer' else GRUClassifier()).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    lossfn=torch.nn.BCEWithLogitsLoss()
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=float('inf'); stale=0; history=[]; state=None
    if device.type=='cuda': torch.cuda.synchronize()
    started=time.perf_counter()
    for epoch in range(args.epochs):
        losses={}
        for phase in ['train','val']:
            model.train(phase=='train'); total=0
            for x,mask,y in loaders[phase]:
                x,mask,y=x.to(device),mask.to(device),y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase=='train'),torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    loss=lossfn(model(x,mask),y)
                if phase=='train':
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1); scaler.step(optimizer); scaler.update()
                total+=loss.item()*len(y)
            losses[phase]=total/len(loaders[phase].dataset)
        history.append(losses); print(epoch+1,losses,flush=True)
        if losses['val']<best:
            best=losses['val']; stale=0; state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
        else: stale+=1
        if stale>=7: break
    if device.type=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    model.load_state_dict(state); model.eval()
    def predict(split):
        probs=[]; targets=[]
        with torch.no_grad():
            for x,m,y in loaders[split]:
                probs.extend(torch.sigmoid(model(x.to(device),m.to(device))).cpu().tolist()); targets.extend(y.tolist())
        return targets,probs
    y,p=predict('val'); threshold=tune_threshold(y,p)
    dataset_hash=hashlib.sha256((paths['processed']/'dataset.pt').read_bytes()).hexdigest()
    checkpoint={'state_dict':state,'architecture':architecture,'stats':data['stats'],'threshold':threshold,'target_days':data['target_days'],'model':args.model,'version':args.run or 'research-v1','cohort_size':len(data['y']),'dataset_sha256':dataset_hash,'training_config':vars(args),'seed':SEED}
    torch.save(checkpoint,checkpoint_path)
    result={'validation':metrics(y,p,threshold),'history':history,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'training_seconds':elapsed,'epochs_completed':len(history),'cohort_size':len(data['y']),'target_days':data['target_days'],'limitations':'Small single-center survivor-only feasibility experiment; no clinical validation'}
    # One held-out evaluation per training run; do not tune using test results.
    y,p=predict('test'); result['test']=metrics(y,p,threshold)
    result['test']['bootstrap_95_ci']=bootstrap_intervals(y,p,[data['cohort'][i]['subjectid'] for i in data['splits']['test']])
    result['run']=args.run; result['checkpoint']=str(checkpoint_path.relative_to(ROOT))
    result.update(training_config=vars(args),seed=SEED,torch_version=torch.__version__,selected_epoch=min(range(len(history)),key=lambda i:history[i]['val'])+1,dataset_sha256=dataset_hash,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
    save_error_analysis(args.model,data,data['splits']['test'],p,threshold,paths['results']/'errors')
    (paths['results']/f'metrics/{args.model}.json').write_text(json.dumps(result,indent=2))
    from utils.visualization import evaluation_plots
    evaluation_plots(args.model,history,y,p,result['test'],paths['results']/'plots')
if __name__=='__main__': main()
