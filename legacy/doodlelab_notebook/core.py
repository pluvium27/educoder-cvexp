"""离线涂鸦实验接口。图片统一为黑底亮线，float32 [N,1,28,28]、范围[0,1]。

训练只使用本地文件；不会下载数据或模型。修改网络时保留输入和8维logits输出。
"""
from __future__ import annotations
import copy, hashlib, json, os, platform, random, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CODE_VERSION = '1.2.0'
PREPROCESS = {'version':'external-v1', 'polarity':'bright_ink_dark_background',
              'shape':[1,28,28], 'range':[0,1], 'fit_size':26,
              'threshold':40, 'centering':'bounding_box', 'photo_autocontrast':True}
DEFAULT_CONFIG = dict(model='mlp', hidden_sizes=[64,32], channels=[8,16], dropout=0.0,
    lr=0.001, epochs=12, batch_size=64, weight_decay=0.0, seed=20261008,
    threads=2, device='cpu', train_split='train_basic', augment_shift=0, custom_params={})

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''): h.update(block)
    return h.hexdigest()

def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

@dataclass
class DatasetBundle:
    """x float32 [N,1,28,28]；y int64 [N]；labels 固定中英文映射；meta 来源。"""
    x: torch.Tensor
    y: torch.Tensor
    labels: list
    meta: dict
    source_index: np.ndarray
    sample_hash: np.ndarray
    def __len__(self): return len(self.y)

def load_dataset(data_dir, split='train_basic', verify=True):
    """输入目录Path/str、划分名str，返回DatasetBundle。默认校验文件SHA256。

    split: train_basic/train_extended/val_basic/challenge_original/shift/thick/occlusion
    （挑战名必须带challenge_前缀）。本学生包已含train_extended，默认使用基础训练。
    """
    root=Path(data_dir)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if split not in manifest['splits']: raise ValueError(f'未知划分: {split}')
    info=manifest['splits'][split]; file=root/info['file']
    if not file.exists():
        raise FileNotFoundError(f'{file} 不存在。扩展文件缺失时请重新完整解压学生包；默认基础路线用train_basic。')
    if verify and sha256(file)!=info['sha256']: raise ValueError(f'{file.name} 校验失败，请重新解压课程发布包。')
    with np.load(file, allow_pickle=False) as z:
        images=z['images']; targets=z['labels']; idx=z['source_index'].copy(); hashes=z['sample_hash'].copy()
    if images.dtype!=np.uint8 or images.shape!=(len(targets),28,28): raise ValueError('图像类型/形状错误')
    if len(targets)==0 or targets.min()<0 or targets.max()>=len(manifest['labels']): raise ValueError('标签越界')
    if targets.dtype!=np.int64: raise ValueError('标签必须为int64')
    x=torch.from_numpy(images.copy()).unsqueeze(1).float()/255.0
    return DatasetBundle(x,torch.from_numpy(targets.copy()),manifest['labels'],
        dict(data_version=manifest['data_version'], split=split, sha256=info['sha256'],
             manifest_sha256=sha256(root/'manifest.json'), count=len(targets), preprocessing=manifest['preprocessing']),idx,hashes)

def set_seed(seed, threads=2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)

def resolve_config(config=None):
    cfg=copy.deepcopy(DEFAULT_CONFIG)
    if config:
        unknown=set(config)-set(cfg)
        if unknown: raise ValueError(f'未知参数（检查拼写）: {unknown}')
        cfg.update(copy.deepcopy(config))
    if cfg['model'] not in ('mlp','cnn','custom'): raise ValueError('model选择mlp、cnn或custom')
    if not isinstance(cfg['seed'],int) or not 0<=cfg['seed']<2**32: raise ValueError('seed需0—2³²−1整数')
    if not isinstance(cfg['custom_params'],dict): raise ValueError('custom_params需字典')
    json.dumps(cfg['custom_params'],allow_nan=False)
    for key in ('epochs','batch_size','threads'):
        if not isinstance(cfg[key],int) or cfg[key]<1: raise ValueError(f'{key}必须为正整数')
    if not np.isfinite(cfg['lr']) or cfg['lr']<=0: raise ValueError('lr必须为有限正数')
    if not np.isfinite(cfg['dropout']) or not np.isfinite(cfg['weight_decay']) or not 0<=cfg['dropout']<1 or cfg['weight_decay']<0: raise ValueError('dropout/weight_decay范围错误')
    if cfg['device']!='cpu': raise ValueError('本教学实现固定CPU；进阶设备需另行修改并记录环境')
    if cfg['train_split'] not in ('train_basic','train_extended'): raise ValueError('验证资源不得用于训练')
    if cfg['augment_shift'] not in (0,1,2,3): raise ValueError('augment_shift为0—3整数像素')
    for key in ('hidden_sizes','channels'):
        if not cfg[key] or not all(isinstance(v,int) and v>0 for v in cfg[key]): raise ValueError(f'{key}需正整数列表')
    if len(cfg['channels'])!=2: raise ValueError('CNN需要两个channels值')
    return cfg

def build_model(config=None, num_classes=8):
    """config字典→nn.Module。输入[B,1,28,28]，输出[B,8]未softmax的logits。

    默认MLP:784→64→32→8，ReLU。CNN:两次3×3卷积/池化→32→8。
    初始化在此按seed固定；与train_model的批次seed配合可重复。
    """
    cfg=resolve_config(config); set_seed(cfg['seed'],cfg['threads'])
    if cfg['model']=='mlp':
        layers=[nn.Flatten()]; prev=784
        for width in cfg['hidden_sizes']:
            layers += [nn.Linear(prev,width),nn.ReLU(),nn.Dropout(cfg['dropout'])]; prev=width
        layers.append(nn.Linear(prev,num_classes))
    elif cfg['model']=='cnn':
        a,b=cfg['channels']
        layers=[nn.Conv2d(1,a,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),
            nn.Linear(b*7*7,32),nn.ReLU(),nn.Dropout(cfg['dropout']),nn.Linear(32,num_classes)]
    if cfg['model']=='custom':
        from .custom_model import build_custom_model
        model=build_custom_model(copy.deepcopy(cfg['custom_params']),num_classes)
        if not isinstance(model,nn.Module): raise TypeError('build_custom_model必须返回torch.nn.Module')
        model.to('cpu'); model.eval()
        with torch.no_grad():
            output=model(torch.zeros(2,1,28,28))
        if output.shape!=(2,num_classes) or not torch.isfinite(output).all():
            raise ValueError('自定义模型应接受[B,1,28,28]，返回有限的[B,8] logits')
        model.train()
    else:
        model=nn.Sequential(*layers)
    model.doodle_config=cfg
    return model.to('cpu')

def _shift_batch(x, amount):
    if not amount: return x
    out=torch.zeros_like(x)
    shifts=torch.randint(-amount,amount+1,(len(x),2))
    for i,(dy,dx) in enumerate(shifts.tolist()):
        sy0,sy1=max(0,-dy),min(28,28-dy); sx0,sx1=max(0,-dx),min(28,28-dx)
        out[i,:,sy0+dy:sy1+dy,sx0+dx:sx1+dx]=x[i,:,sy0:sy1,sx0:sx1]
    return out

def evaluate_model(model, dataset, batch_size=256):
    """模型+DatasetBundle→dict。accuracy为0—1；混淆矩阵行真实、列预测。

    返回loss/count/per_class/confusion/predictions/true_labels/scores/error_indices。
    默认CPU、eval和无梯度；scores是softmax分数，不保证正确概率。
    """
    was_training=model.training; model.eval()
    scores=[]; loss=0.0
    try:
        with torch.no_grad():
            for x,y in DataLoader(TensorDataset(dataset.x,dataset.y),batch_size=batch_size,shuffle=False,num_workers=0):
                logits=model(x)
                if logits.shape!=(len(y),len(dataset.labels)): raise ValueError('模型输出应为[B,8]')
                if not torch.isfinite(logits).all(): raise ValueError('模型出现非有限输出')
                loss+=nn.functional.cross_entropy(logits,y,reduction='sum').item()
                scores.append(torch.softmax(logits,dim=1).numpy())
    finally: model.train(was_training)
    scores=np.concatenate(scores); pred=scores.argmax(1); true=dataset.y.numpy()
    confusion=np.zeros((len(dataset.labels),len(dataset.labels)),dtype=np.int64)
    np.add.at(confusion,(true,pred),1)
    per_class=[]
    for i,label in enumerate(dataset.labels):
        n=int(confusion[i].sum()); correct=int(confusion[i,i])
        per_class.append(dict(**label,count=n,correct=correct,accuracy=correct/n if n else None))
    return dict(accuracy=float((pred==true).mean()), loss=loss/len(true),count=len(true),
        per_class=per_class,confusion=confusion.tolist(),predictions=pred.tolist(),
        true_labels=true.tolist(),scores=scores.tolist(),error_indices=np.flatnonzero(pred!=true).tolist(),
        dataset=dataset.meta)

def train_model(model, train_data, val_data, config=None, verbose=True):
    """输入新模型、训练/验证Bundle、config→(model,result)。默认训练12轮。

    每轮完整记录训练集与验证集loss/accuracy。Adam + CrossEntropyLoss。
    保留最后一轮权重，不自动挑选最高验证分；修改前先保存旧实验。
    """
    cfg=resolve_config(config or getattr(model,'doodle_config',{}))
    built=getattr(model,'doodle_config',None)
    if built and any(cfg[k]!=built[k] for k in ('model','hidden_sizes','channels','dropout','custom_params')):
        raise ValueError('模型结构配置已改变；请先build_model(config)重建模型再训练')
    if train_data.meta['split']!=cfg['train_split']: raise ValueError('配置与训练划分不一致')
    if val_data.meta['split']!='val_basic': raise ValueError('训练监测统一使用val_basic')
    if train_data.labels!=val_data.labels: raise ValueError('标签映射不同')
    if set(train_data.sample_hash.tolist()) & set(val_data.sample_hash.tolist()): raise ValueError('训练与验证图像重叠')
    model.to('cpu'); set_seed(cfg['seed'],cfg['threads']); model.doodle_config=cfg
    gen=torch.Generator().manual_seed(cfg['seed'])
    loader=DataLoader(TensorDataset(train_data.x,train_data.y),batch_size=cfg['batch_size'],
        shuffle=True,generator=gen,num_workers=0)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    history=[]; started=time.perf_counter(); cpu_start=time.process_time()
    for epoch in range(1,cfg['epochs']+1):
        epoch_start=time.perf_counter(); model.train(); grad_nonzero=False
        for x,y in loader:
            optimizer.zero_grad(); logits=model(_shift_batch(x,cfg['augment_shift']))
            loss=nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss): raise ValueError('loss非有限，请核查输入/学习率')
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise ValueError('梯度非有限')
                    grad_nonzero |= bool(torch.count_nonzero(p.grad))
            optimizer.step()
        if not grad_nonzero: raise ValueError('所有梯度为零；检查网络与训练代码')
        tr=evaluate_model(model,train_data); va=evaluate_model(model,val_data)
        row=dict(epoch=epoch,train_loss=tr['loss'],train_accuracy=tr['accuracy'],
            val_loss=va['loss'],val_accuracy=va['accuracy'],seconds=time.perf_counter()-epoch_start)
        history.append(row)
        if verbose: print(f"第{epoch:02d}轮 | 训练准确率 {tr['accuracy']:.1%} | 验证准确率 {va['accuracy']:.1%} | 验证损失 {va['loss']:.3f}")
    model.eval()
    result=dict(config=cfg,history=history,seconds=time.perf_counter()-started,
        cpu_seconds=time.process_time()-cpu_start,parameter_count=sum(p.numel() for p in model.parameters()),
        device='cpu',data=dict(train=train_data.meta,validation=val_data.meta),labels=train_data.labels,
        preprocess=copy.deepcopy(PREPROCESS),code_version=CODE_VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                         pillow=Image.__version__,platform=platform.platform(),threads=torch.get_num_threads()),
        training_policy='last_epoch; validation_visible; no_independent_test')
    return model,result

def preprocess_image(image, mode='drawing', polarity='auto', crop=None, threshold=40, fit_size=26):
    """路径str/Path、PIL.Image或uint8数组→dict。

    返回original(PIL RGB)、processed(PIL L黑底亮线28²)、tensor[1,1,28,28]、info。
    drawing:白底深线或透明背景；photo:白纸深笔，可crop=(左,上,右,下)像素。
    polarity=auto/dark/light。自动以边缘明暗猜背景，失败时手动指定。
    threshold 0—254用于寻找墨迹框，不把所有灰度强制二值化；fit_size 8—28。
    """
    if mode not in ('drawing','photo'): raise ValueError('mode选择drawing/photo')
    if polarity not in ('auto','dark','light'): raise ValueError('polarity选择auto/dark/light')
    if not 0<=threshold<255 or not 8<=fit_size<=28: raise ValueError('threshold/fit_size范围错误')
    if isinstance(image,(str,Path)):
        with Image.open(image) as opened: im=ImageOps.exif_transpose(opened).convert('RGBA')
    elif isinstance(image,Image.Image): im=ImageOps.exif_transpose(image).convert('RGBA')
    elif isinstance(image,np.ndarray):
        if image.dtype!=np.uint8: raise TypeError('数组需uint8[0,255]；浮点输入请先明确转换')
        im=Image.fromarray(image).convert('RGBA')
    else: raise TypeError('image需路径、PIL图像或uint8数组')
    # 透明区域合成到白底；否则转灰度会把透明背景变成黑色。
    im=Image.alpha_composite(Image.new('RGBA',im.size,'white'),im).convert('RGB')
    original=im.copy()
    if crop:
        l,t,r,b=map(int,crop)
        if not (0<=l<r<=im.width and 0<=t<b<=im.height): raise ValueError('crop超出图像或为空')
        im=im.crop((l,t,r,b))
    gray=ImageOps.grayscale(im)
    if mode=='photo':
        gray=ImageOps.autocontrast(gray,cutoff=1).filter(ImageFilter.MedianFilter(3))
    a=np.asarray(gray)
    border=np.concatenate([a[0],a[-1],a[:,0],a[:,-1]])
    chosen=polarity if polarity!='auto' else ('dark' if np.median(border)>127 else 'light')
    ink=255-a if chosen=='dark' else a.copy()
    ys,xs=np.where(ink>threshold)
    if not len(xs): raise ValueError('未找到线条：空白图、阈值过高或黑白方向错误')
    bbox=(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1))
    patch=Image.fromarray(ink).crop(bbox)
    scale=fit_size/max(patch.size); size=tuple(max(1,round(v*scale)) for v in patch.size)
    patch=patch.resize(size,Image.Resampling.LANCZOS)
    processed=Image.new('L',(28,28),0); processed.paste(patch,((28-size[0])//2,(28-size[1])//2))
    tensor=torch.from_numpy(np.array(processed,dtype=np.float32)/255.0)[None,None]
    return dict(original=original,processed=processed,tensor=tensor,
        info=dict(version=PREPROCESS['version'],mode=mode,polarity=chosen,crop=list(crop) if crop else None,
                  ink_bbox=list(bbox),threshold=threshold,fit_size=fit_size,centering='bounding_box'))

def predict_image(model, image, labels=None, top_k=3, **preprocess_options):
    """模型+路径/PIL/数组或preprocess_image结果→预测dict（包含预处理图）。

    默认用加载模型自带映射/预处理参数。top_k 1—8。输出标签与softmax分数。
    """
    labels=labels or getattr(model,'doodle_labels',None)
    if labels is None: raise ValueError('请传labels=train_data.labels或先load_model')
    if not 1<=top_k<=len(labels): raise ValueError('top_k越界')
    defaults=getattr(model,'doodle_preprocess',PREPROCESS)
    options=dict(threshold=defaults['threshold'],fit_size=defaults['fit_size']); options.update(preprocess_options)
    prepared=image if isinstance(image,dict) and 'tensor' in image else preprocess_image(image,**options)
    x=prepared['tensor']
    if x.shape!=(1,1,28,28) or not torch.isfinite(x).all() or x.min()<0 or x.max()>1: raise ValueError('预测输入应为[1,1,28,28]、[0,1]')
    model.eval()
    with torch.no_grad(): scores=torch.softmax(model(x),1)[0].numpy()
    order=np.argsort(-scores)[:top_k]
    return dict(top=[dict(**labels[int(i)],score=float(scores[i])) for i in order],scores=scores.tolist(),
                prepared=prepared,notice='输出分数不构成正确性保证；类别外输入仍会被归入八类之一。')

def save_experiment(output_dir, model, result, evaluations=None):
    """新目录、模型、训练结果、可选评价dict → 目录Path；拒绝覆盖。

    默认只写model.pt与实验记录.txt。设置、逐轮指标、标签和预处理存入权重包；
    文字记录可直接复制给AI，不写逐图JSON/CSV。自定义模型另留构建函数源码。
    保存最后一轮模型，不提供断点续训。
    """
    from .records import compact_evaluation, write_log
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=False)
    record=json.loads(json.dumps(result,ensure_ascii=False,allow_nan=False))
    record['evaluations']={name:compact_evaluation(ev) for name,ev in (evaluations or {}).items()}
    if record['config']['model']=='custom':
        import shutil
        source=output/'source/doodlelab';source.mkdir(parents=True)
        file=Path(__file__).parent/'custom_model.py'
        shutil.copy2(file,source/file.name)
        record['source_sha256']={file.name:sha256(file)}
    torch.save(dict(state_dict=model.state_dict(),config=record['config'],labels=record['labels'],
                    preprocess=record['preprocess'],code_version=CODE_VERSION,record=record),output/'model.pt')
    write_log(output,record,record['evaluations'])
    model.doodle_labels=record['labels'];model.doodle_preprocess=record['preprocess']
    return output

def load_model(path):
    """实验目录或model.pt路径→(CPU eval模型,metadata)。weights_only=True读取。

    新版设置保存在model.pt中；兼容旧版experiment.json目录与权重校验。配置重建结构后加载state_dict。
    model=custom时自动调用custom_model.py的build_custom_model重建结构。
    若其源码已改变，需先恢复该实验source/doodlelab/custom_model.py并重启内核。
    """
    path=Path(path); metadata=None
    if path.is_dir():
        file=path/'model.pt'
        legacy=path/'experiment.json'
        if legacy.exists():
            metadata=json.loads(legacy.read_text(encoding='utf-8'))
            if sha256(file)!=metadata['model_sha256']:raise ValueError('权重校验失败')
    else: file=path
    checkpoint=torch.load(file,map_location='cpu',weights_only=True)
    metadata=metadata or checkpoint.get('record')
    if checkpoint['config']['model']=='custom' and metadata:
        expected=metadata.get('source_sha256',{}).get('custom_model.py')
        if expected and sha256(Path(__file__).parent/'custom_model.py')!=expected:
            raise ValueError('自定义模型源码已改变：请从该实验source/doodlelab恢复custom_model.py，再重启内核加载。')
    model=build_model(checkpoint['config'],len(checkpoint['labels']))
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    model.doodle_labels=checkpoint['labels']; model.doodle_preprocess=checkpoint['preprocess']
    return model,metadata or {k:v for k,v in checkpoint.items() if k!='state_dict'}
