import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay,PrecisionRecallDisplay
from config import ROOT

def evaluation_plots(name,history,y,p,result,output=None):
    output=output or ROOT/'results/plots'
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    RocCurveDisplay.from_predictions(y,p,ax=axes[0]); PrecisionRecallDisplay.from_predictions(y,p,ax=axes[1])
    axes[2].plot([0,1],[0,1],'--'); axes[2].plot(result['calibration']['predicted'],result['calibration']['observed'],'o-'); axes[2].set(xlabel='Predicted risk',ylabel='Observed fraction')
    fig.tight_layout(); fig.savefig(output/f'{name}_evaluation.png'); plt.close(fig)
    if history:
        fig,ax=plt.subplots(); ax.plot([r['train'] for r in history],label='Train'); ax.plot([r['val'] for r in history],label='Validation'); ax.legend(); ax.set(xlabel='Epoch',ylabel='BCE loss'); fig.savefig(output/f'{name}_training.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(4,4)); matrix=result['confusion_matrix']; ax.imshow(matrix,cmap='Blues')
    for row in range(2):
        for col in range(2): ax.text(col,row,str(matrix[row][col]),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Not prolonged','Prolonged'],yticklabels=['Not prolonged','Prolonged'],xlabel='Predicted',ylabel='Actual',title=name+' held-out test'); fig.tight_layout(); fig.savefig(output/f'{name}_confusion.png'); plt.close(fig)
