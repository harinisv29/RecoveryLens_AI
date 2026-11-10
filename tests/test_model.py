import torch
from models.temporal_transformer import TemporalTransformer
from models.gru_baseline import GRUClassifier
from inference.predict import load_checkpoint

def test_forward_and_padding():
    torch.manual_seed(42)
    model=TemporalTransformer().eval()
    x=torch.randn(2,10,14); mask=torch.ones(2,10,dtype=torch.bool); mask[0,5:]=False
    y=model(x,mask); x[0,5:]=100
    assert y.shape==(2,) and torch.allclose(y,model(x,mask),atol=1e-5)
    assert GRUClassifier()(x,mask).shape==(2,)

def test_checkpoint_roundtrip(tmp_path):
    # Random weights verify serialization only; never used as application predictions.
    model=TemporalTransformer(); path=tmp_path/'test.pt'
    torch.save({'architecture':{},'state_dict':model.state_dict()},path)
    restored,_=load_checkpoint(path)
    assert not restored.training
