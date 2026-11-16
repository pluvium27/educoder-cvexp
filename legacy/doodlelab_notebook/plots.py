"""图表用英文类别名避免学生电脑缺中文字体；解释文字在Notebook中。"""
import numpy as np
import matplotlib.pyplot as plt

def show_samples(data, per_class=4):
    fig,axes=plt.subplots(len(data.labels),per_class,figsize=(per_class*1.5,len(data.labels)*1.25),squeeze=False)
    for i,label in enumerate(data.labels):
        rows=np.flatnonzero(data.y.numpy()==i)[:per_class]
        for j,ax in enumerate(axes[i]):
            ax.axis('off')
            if j<len(rows): ax.imshow(data.x[rows[j],0],cmap='gray',vmin=0,vmax=1)
            if j==0: ax.set_title(f"{i}: {label['en']}",fontsize=9)
    fig.tight_layout(); return fig

def plot_history(history):
    fig,axes=plt.subplots(1,2,figsize=(10,3)); epochs=[r['epoch'] for r in history]
    for key,label in [('train','train'),('val','validation')]:
        axes[0].plot(epochs,[r[key+'_loss'] for r in history],label=label)
        axes[1].plot(epochs,[r[key+'_accuracy'] for r in history],label=label)
    for ax,title in zip(axes,['Loss (lower is better)','Accuracy']):
        ax.set(xlabel='Epoch',title=title); ax.legend(); ax.grid(alpha=.2)
    axes[1].set_ylim(0,1); fig.tight_layout(); return fig

def plot_confusion(evaluation, labels):
    cm=np.asarray(evaluation['confusion']); fig,ax=plt.subplots(figsize=(7,6))
    ax.imshow(cm,cmap='Blues')
    ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=[x['en'] for x in labels],
           yticklabels=[x['en'] for x in labels],xlabel='Predicted',ylabel='True',title='Confusion counts')
    for i in range(len(labels)):
        for j in range(len(labels)): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>cm.max()*.55 else 'black')
    plt.setp(ax.get_xticklabels(),rotation=40,ha='right'); fig.tight_layout(); return fig

def show_errors(data, evaluation, limit=12):
    rows=evaluation['error_indices'][:limit]
    if not rows: print('当前验证集无错误样例；结论仍需结合数据独立性和其他输入评价。'); return None
    fig,axes=plt.subplots(int(np.ceil(len(rows)/4)),4,figsize=(8,2*int(np.ceil(len(rows)/4))),squeeze=False)
    for ax in axes.flat: ax.axis('off')
    for ax,i in zip(axes.flat,rows):
        t=evaluation['true_labels'][i]; p=evaluation['predictions'][i]
        ax.imshow(data.x[i,0],cmap='gray',vmin=0,vmax=1)
        ax.set_title(f"row {i}: {data.labels[t]['en']}\n→ {data.labels[p]['en']}",fontsize=8)
    fig.tight_layout(); return fig

def show_prediction(prediction):
    p=prediction['prepared']; fig,axes=plt.subplots(1,2,figsize=(7,3))
    axes[0].imshow(p['original']); axes[0].set_title('Original input')
    axes[1].imshow(p['processed'],cmap='gray',vmin=0,vmax=255); axes[1].set_title('Model sees: 28 x 28')
    for ax in axes: ax.axis('off')
    fig.tight_layout(); return fig
