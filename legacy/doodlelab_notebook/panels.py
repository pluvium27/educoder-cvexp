"""本地ipywidgets操作面板。滑动只改设置；耗时操作必须按按钮。

没有控件组件时保留图片、默认训练和普通函数入口；不访问CDN或外部服务。
返回控件字典可检查实际配置与触发回调；学生通常不需要读本文件。
"""
from pathlib import Path
import numpy as np
from IPython.display import display, clear_output
from .studio import _show, _esc, _image, _figure
from . import core, plots


def _widgets():
    try:
        import ipywidgets as w
        return w
    except ImportError:
        _show('<p>交互控件组件尚未安装。默认训练仍可运行；调参可用lab.train("实验名", lr=0.0003)，图片可用lab.predict("my_images/doodle.png")。要显示按钮，请按资源/requirements-common.txt预先安装ipywidgets并重启Jupyter。</p>')
        return None


def _remember(lab, name, controls):
    if not hasattr(lab,'panels'): lab.panels={}
    lab.panels[name]=controls
    return controls


def _action(button, output, operation):
    """同步回调：防重复点击，错误以可读文字显示，恢复按钮状态。"""
    button.disabled=True
    try:
        with output:
            clear_output(wait=True)
            try: operation()
            except Exception as error:
                _show('<p style="color:#a33"><b>操作未完成：</b>'+_esc(error)+'<br>检查上方设置后再按按钮。已保存的旧实验不会被覆盖。</p>')
    finally:
        button.disabled=False


def guessing_panel(lab):
    w=_widgets(); data=lab.dataset('val_basic')
    rng=np.random.default_rng(42); state={'row':int(rng.integers(len(data)))}
    if w is None:
        i=state['row']; _show(_image(data.x[i,0],224)+'<details><summary>展开查看原数据标注</summary>'+_esc(lab.labels[int(data.y[i])]['zh'])+'</details>'); return
    picture=w.Output(); feedback=w.Output()
    choice=w.Dropdown(options=[(x['zh'],x['id']) for x in lab.labels],description='我猜是')
    reveal=w.Button(description='揭晓标注',button_style='info',icon='eye')
    another=w.Button(description='换一张',icon='refresh')
    def redraw():
        with picture:
            clear_output(wait=True); _show(_image(data.x[state['row'],0],224))
        feedback.clear_output()
    def answer(_):
        i=state['row']; truth=int(data.y[i])
        with feedback:
            clear_output(wait=True)
            message='所选类别与原数据标注一致。' if choice.value==truth else '所选类别与原数据标注不同，可分析图案中的混淆线索。'
            _show('<p>原数据标注：<b>'+_esc(lab.labels[truth]['zh'])+'</b>。'+message+'<br>样本来自基础验证集，未进入训练；分析时可讨论图案与标注的歧义。</p>')
    def next_picture(_):
        old=state['row']
        state['row']=(old+int(rng.integers(1,len(data))))%len(data); redraw()
    reveal.on_click(answer); another.on_click(next_picture)
    redraw(); display(w.VBox([picture,w.HBox([choice,reveal,another]),feedback]))
    return _remember(lab,'play',dict(choice=choice,reveal=reveal,another=another,state=state,feedback=feedback))


def gallery_panel(lab):
    w=_widgets()
    if w is None: return
    category=w.Dropdown(options=[(x['zh'],x['id']) for x in lab.labels],description='多看一种')
    output=w.Output(); data=lab.dataset()
    def preview(change=None):
        rows=np.flatnonzero(data.y.numpy()==category.value)[2:8]
        with output:
            clear_output(wait=True)
            _show('<div class="dl-grid">'+''.join('<div class="dl-card">'+_image(data.x[i,0],90)+'</div>' for i in rows)+'</div>')
    category.observe(preview,names='value')
    preview(); display(w.VBox([category,output]))
    return _remember(lab,'gallery',dict(category=category,output=output))


def experiment_panel(lab):
    w=_widgets()
    if w is None: return
    style={'description_width':'110px'}; layout=w.Layout(width='100%',max_width='360px')
    name=w.Text(value='学习率对照',description='实验名',style=style,layout=layout)
    model=w.Dropdown(options=[('小MLP（基础）','mlp'),('小CNN（进阶）','cnn')],description='模型类型',style=style,layout=layout)
    lr=w.SelectionSlider(options=[('0.0001',.0001),('0.0003',.0003),('0.001',.001),('0.003',.003),('0.01',.01)],value=.0003,description='每步学习幅度',continuous_update=False,style=style,layout=layout)
    epochs=w.IntSlider(value=12,min=4,max=30,step=2,description='完整学习轮数',continuous_update=False,style=style,layout=layout)
    hidden=w.Dropdown(options=[('32 → 16（更小）',(32,16)),('64 → 32（默认）',(64,32)),('128 → 64',(128,64)),('256 → 128',(256,128))],value=(64,32),description='MLP两层大小',style=style,layout=layout)
    channels=w.Dropdown(options=[('8 → 16（默认）',(8,16)),('16 → 32',(16,32))],value=(8,16),description='CNN通道数',style=style,layout=layout,disabled=True)
    train=w.Dropdown(options=[('每类300张（基础）','train_basic'),('每类3000张（进阶）','train_extended')],description='训练资料',style=style,layout=layout)
    seed=w.IntText(value=20261008,description='随机种子',style=style,layout=layout)
    dropout=w.FloatSlider(value=0,min=0,max=.5,step=.1,description='dropout',continuous_update=False,style=style,layout=layout)
    shift=w.IntSlider(value=0,min=0,max=3,description='训练随机偏移',continuous_update=False,style=style,layout=layout)
    advanced=w.Accordion(children=[w.VBox([train,seed,channels,dropout,shift])]); advanced.set_title(0,'进阶设置（可以跳过）'); advanced.selected_index=None
    go=w.Button(description='从头训练并保存',button_style='success',icon='play',layout=w.Layout(width='220px'))
    settings=w.HTML(); output=w.Output()
    controls=dict(name=name,model=model,lr=lr,epochs=epochs,hidden_sizes=hidden,channels=channels,train_split=train,seed=seed,dropout=dropout,augment_shift=shift,go=go,output=output)
    def config():
        return dict(model=model.value,lr=lr.value,epochs=epochs.value,hidden_sizes=list(hidden.value),channels=list(channels.value),train_split=train.value,seed=seed.value,dropout=dropout.value,augment_shift=shift.value)
    def changed(change=None):
        hidden.disabled=model.value!='mlp'; channels.disabled=model.value!='cnn'
        actual=core.resolve_config(config())
        structure=actual['hidden_sizes'] if model.value=='mlp' else actual['channels']
        settings.value='<p><b>准备执行：</b>'+_esc(model.value)+f' · 结构 {_esc(structure)} · lr={lr.value} · {epochs.value}轮 · {_esc(train.value)} · seed={seed.value}<br>所有实验固定使用同一基础验证集；改动尚未生效，请按训练按钮。</p>'
    for control in [model,lr,epochs,hidden,channels,train,seed,dropout,shift]: control.observe(changed,names='value')
    def operation():
        lab.train(name.value,**config())
        _show(_figure(plots.plot_history(lab.latest.result['history'])))
        lab.compare()
        _show('<p>新实验已成为当前模型。查看其完整结果：重新运行第4部分的lab.results()；识别图片：回到第5部分，保持同一文件和处理设置，再按预测按钮。此前显示的输出不会自动更新，请核对结果中的实验名称。</p>')
    go.on_click(lambda _: _action(go,output,operation))
    changed()
    display(w.VBox([w.HTML('<p>请先确定研究问题，再修改一个主要条件。学习幅度对应lr；轮数对应epochs。基础对照建议保持进阶设置不变。</p>'),name,model,lr,epochs,hidden,advanced,settings,go,output]))
    return _remember(lab,'experiment',controls)


def image_panel(lab):
    w=_widgets()
    if w is None: lab.predict('资源/examples/synthetic_transparent_apple.png'); return
    choices=[('随包例图：透明画板苹果（合成）','资源/examples/synthetic_transparent_apple.png'),('随包例图：纸面苹果（合成示意）','资源/examples/synthetic_paper_apple.png')]
    for path in sorted((lab.root/'my_images').glob('*')):
        if path.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.bmp'):
            choices.append(('本组图片：'+path.name,str(path.relative_to(lab.root))))
    select=w.Dropdown(options=choices,description='选一张',layout=w.Layout(width='100%',max_width='550px'))
    path=w.Text(value=choices[0][1],description='文件路径',layout=w.Layout(width='100%',max_width='550px'))
    mode=w.Dropdown(options=[('画板/图片','drawing'),('白纸深色笔照片','photo')],description='输入类型')
    polarity=w.Dropdown(options=[('自动判断','auto'),('白底深色线','dark'),('黑底亮色线','light')],description='黑白方向')
    threshold=w.IntSlider(value=40,min=0,max=200,step=10,description='找线条阈值',continuous_update=False)
    fit=w.IntSlider(value=26,min=8,max=28,description='图案占几格',continuous_update=False)
    crop_enabled=w.Checkbox(value=False,description='启用裁剪（先看原图尺寸）')
    crop=[w.IntText(value=0,description=x,layout=w.Layout(width='175px'),style={'description_width':'30px'}) for x in ['左','上','右','下']]
    advanced=w.Accordion(children=[w.VBox([mode,polarity,threshold,fit,crop_enabled,w.VBox(crop)])]); advanced.set_title(0,'图片处理设置（照片建议先裁剪）'); advanced.selected_index=None
    go=w.Button(description='让当前模型猜',button_style='info',icon='image')
    output=w.Output()
    def selected(change):
        path.value=change['new']; mode.value='photo' if 'paper' in path.value else 'drawing'
    select.observe(selected,names='value')
    def operation():
        lab.predict(path.value,mode=mode.value,polarity=polarity.value,threshold=threshold.value,fit_size=fit.value,crop=tuple(x.value for x in crop) if crop_enabled.value else None)
    go.on_click(lambda _: _action(go,output,operation))
    display(w.VBox([w.HTML('<p>画板保存PNG后放进my_images，再在文件路径填入my_images/文件名.png。新加入的文件也可直接填写路径。按钮使用最近训练或加载的模型；先查看28×28图是否保留了主要线索。</p>'),select,path,advanced,go,output]))
    _action(go,output,operation)
    return _remember(lab,'image',dict(select=select,path=path,mode=mode,polarity=polarity,threshold=threshold,fit_size=fit,crop_enabled=crop_enabled,crop=crop,go=go,output=output))


def challenge_panel(lab):
    w=_widgets()
    if w is None:
        _show('<p>需要时运行lab.challenge()；默认不执行进阶评价。</p>'); return
    go=w.Button(description='评价当前模型的挑战表现',icon='search',layout=w.Layout(width='280px'))
    output=w.Output(); go.on_click(lambda _: _action(go,output,lab.challenge))
    display(w.VBox([go,output]))
    return _remember(lab,'challenge',dict(go=go,output=output))
