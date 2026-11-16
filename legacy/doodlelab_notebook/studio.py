"""Notebook的短调用入口：组织实验、显示证据、自动保存，不改变core训练规则。

start_lab(root='.') → DoodleLab。root必须是包含资源/data/与资源/doodlelab/的目录。
lab.train(name, **参数)每次新建模型；lab.latest保存最近一次实验。
扩展开发可沿用core的八个底层接口；交互控件单独放在panels.py。
"""
from __future__ import annotations
import base64
import html
import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import HTML, display
from . import core, plots, records


def _esc(value):
    return html.escape(str(value))


def _png(image):
    stream=io.BytesIO(); image.save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')


def _figure(fig):
    stream=io.BytesIO(); fig.savefig(stream,format='png',dpi=110,bbox_inches='tight')
    plt.close(fig)
    return '<img class="dl-figure" src="data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')+'">'


def _image(tensor, size=128):
    a=np.asarray(tensor); a=np.clip(a*255,0,255).astype('uint8')
    return f'<img style="width:{size}px;image-rendering:pixelated;border-radius:10px" src="{_png(Image.fromarray(a))}">'


def _table(headers, rows):
    width='min-width:850px' if len(headers)>5 else 'min-width:280px'
    return '<div style="overflow:auto"><table style="'+width+'"><thead><tr>'+''.join('<th>'+_esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+_esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def _show(body):
    display(HTML('<div class="doodle-lab">'+body+'</div>'))


@dataclass(repr=False)
class Experiment:
    name: str
    model: object
    result: dict
    evaluations: dict
    path: Path
    reload_difference: float

    def __repr__(self):
        return f"Experiment({self.name!r}, model={self.result['config']['model']!r}, path={str(self.path)!r})"


class Prediction(dict):
    """仍可按dict读取全部预测；Notebook不会自动打印784个像素数字。"""
    def __repr__(self):
        return f"Prediction(top={self['top']!r})"


class DoodleLab:
    """本地CPU工作台；实例间互不共用模型。数据只从root/资源/data读取。

    train返回Experiment；results/compare/challenge等以中文图表显示结果。
    predict返回core.predict_image的dict，显示原图和28×28图，将设置和预测追加到文字记录。
    load返回Experiment并切换当前模型；load后predict不需重新训练。
    """
    def __init__(self, root='.'):
        self.root=Path(root).resolve()
        self.data_dir=self.root/'data'; self.cache={}; self.latest=None
        self.manifest=json.loads((self.data_dir/'manifest.json').read_text(encoding='utf-8'))
        self.labels=self.manifest['labels']

    def dataset(self, split='train_basic'):
        """划分名str → 已校验的DatasetBundle；缓存只避免重复读取，不改数据。"""
        if split not in self.cache: self.cache[split]=core.load_dataset(self.data_dir,split)
        return self.cache[split]

    def _require(self):
        if self.latest is None: raise RuntimeError('请先运行“训练初始模型”，或用lab.load("runs/实验目录")加载模型。')
        return self.latest

    def play(self):
        """猜一张验证涂鸦，选标签、揭晓、换图；不记录人类分数。"""
        from .panels import guessing_panel
        return guessing_panel(self)

    def gallery(self):
        """八类真实训练涂鸦，每类两张；另可下拉查看更多。"""
        data=self.dataset(); cards=[]
        for label in self.labels:
            rows=np.flatnonzero(data.y.numpy()==label['id'])[:2]
            cards.append('<div class="dl-card"><b>'+_esc(label['zh'])+'</b><br>'+''.join(_image(data.x[i,0],94) for i in rows)+'</div>')
        _show('<div class="dl-grid">'+''.join(cards)+'</div><p>每幅图只有28×28个灰度像素。黑底亮线是模型的统一输入；同一类别的画法存在差异。</p>')
        from .panels import gallery_panel
        gallery_panel(self)

    def train(self, name='第一次训练', **parameters):
        """name实验名str，参数同core.DEFAULT_CONFIG；返回Experiment并切换当前模型。

        默认：MLP两隐藏层[64,32]，lr=.001，12轮，基础训练2400张、CPU2线程。
        示例：lab.train('小步学习',lr=.0003)；lab.train('卷积尝试',model='cnn')。
        每次从头初始化，沿用固定验证集；自动保存、重载、比较全部验证分数。
        参数可调lr/epochs/hidden_sizes/channels/dropout/weight_decay/seed/batch_size/
        train_split/augment_shift/custom_params；拒绝拼错的键与验证集训练。
        """
        if not isinstance(name,str) or not re.fullmatch(r'[\w \-]{1,40}',name):
            raise ValueError('实验名用1—40个汉字、字母、数字、空格或下划线，不含路径符号。')
        cfg=core.resolve_config(parameters)
        train=self.dataset(cfg['train_split']); val=self.dataset('val_basic')
        _show(f'<p>⏳ 正在训练“{_esc(name)}”：{len(train)}张训练图，{cfg["epochs"]}轮，CPU。每次都从头训练；请等本单元格运行完成。</p>')
        started=time.perf_counter()
        model=core.build_model(cfg)
        model,result=core.train_model(model,train,val,cfg,verbose=False)
        result['experiment_name']=name
        ev=core.evaluate_model(model,val)
        path=self.root/'runs'/(name.replace(' ','_')+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        core.save_experiment(path,model,result,{'val_basic':ev})
        reloaded,_=core.load_model(path)
        repeat=core.evaluate_model(reloaded,val)
        difference=float(np.max(np.abs(np.asarray(ev['scores'])-np.asarray(repeat['scores']))))
        if difference>1e-6: raise RuntimeError(f'保存、加载后的分数差异为{difference}，请先核查再继续。')
        self.latest=Experiment(name,reloaded,result,{'val_basic':ev},path,difference)
        with (path/'实验记录.txt').open('a',encoding='utf-8') as f:
            f.write(f'\n保存后重载核查：预测分数最大差异{difference:g}。\n')
        fig=plots.plot_history(result['history'])
        fig.savefig(path/'训练曲线.png',dpi=90,bbox_inches='tight');plt.close(fig)
        self.compare(show=False)
        _show(f'<div class="dl-banner"><b>✅ 模型训练完成</b><br>基础验证：{ev["accuracy"]:.1%} · 模型参数：{result["parameter_count"]:,} · 训练耗时：{result["seconds"]:.2f}秒<br>保存、加载分数最大差异：{difference:g}<br>实验目录：<code>{_esc(path.relative_to(self.root))}</code></div><p>保存与核查全过程本次耗时{time.perf_counter()-started:.2f}秒。结果是公开验证集表现；请继续查看类别表现与错误样例。</p>')
        return self.latest

    def results(self):
        """显示最近模型的曲线、每类结果、混淆和错误；图片保存到当前实验目录。"""
        run=self._require(); ev=run.evaluations['val_basic']
        curve=_figure(plots.plot_history(run.result['history']))
        table=_table(['类别','正确数量 / 总数','准确率'],[(x['zh'],f'{x["correct"]} / {x["count"]}',f'{x["accuracy"]:.1%}') for x in ev['per_class']])
        confusion=_figure(plots.plot_confusion(ev,self.labels))
        _show('<p><b>正在查看：'+_esc(run.name)+'</b>。左图loss表示损失（犯错程度）；右图accuracy表示准确率（正确识别比例）。train对应训练图，validation为未进入训练的验证图。</p>'+curve+'<details><summary>展开：八个类别各自表现如何？</summary>'+table+'</details><details><summary>展开：哪些类别容易混淆？</summary><p>行是真实类别，列是预测类别。对角线表示正确识别；其他格子是混淆次数。英文名与上方中文标签一一对应。</p>'+confusion+'</details>')
        self.errors()

    def errors(self, true_label=None, limit=8):
        """true_label=None/0—7；limit正整数。按验证行号显示错误，不挑最好结果。"""
        run=self._require(); ev=run.evaluations['val_basic']; data=self.dataset('val_basic')
        if true_label is not None and true_label not in range(8): raise ValueError('true_label为0—7或None')
        if not isinstance(limit,int) or limit<1: raise ValueError('limit需正整数')
        rows=[i for i in ev['error_indices'] if true_label is None or ev['true_labels'][i]==true_label][:limit]
        cards=[]
        for i in rows:
            t=ev['true_labels'][i]; p=ev['predictions'][i]
            cards.append('<div class="dl-card">'+_image(data.x[i,0])+'<br>标注：'+_esc(self.labels[t]['zh'])+'<br>预测：'+_esc(self.labels[p]['zh'])+f'<small>验证行号 {i} · 输出分数 {max(ev["scores"][i]):.2f}</small></div>')
        _show('<p><b>错误样例分析提示</b>：可分析形状相似、信息简略与关键线索遗漏等可能原因。</p><div class="dl-grid">'+(''.join(cards) or '当前筛选条件下无错误样例。')+'</div>')

    def compare(self, show=True):
        """扫描runs记录，显示设置和基础验证结果；写中文实验汇总.txt，不排行。"""
        rows=[]
        saved=records.saved_runs(self.root)
        for p,r in saved:
            c=r['config']; h=r['history'][-1]
            structure=c.get('custom_params',{}) if c['model']=='custom' else c['hidden_sizes'] if c['model']=='mlp' else c['channels']
            rows.append(dict(name=r.get('experiment_name',p.parent.name),model=c['model'],structure=json.dumps(structure),
                lr=c['lr'],epochs=c['epochs'],train_split=c['train_split'],seed=c['seed'],batch_size=c['batch_size'],
                dropout=c['dropout'],weight_decay=c['weight_decay'],augment_shift=c['augment_shift'],
                train_accuracy=h['train_accuracy'],val_accuracy=h['val_accuracy'],seconds=r['seconds'],path=p.name))
        if rows:
            texts=[]
            for path,record in saved:
                file=path/'实验记录.txt'
                texts.append(file.read_text(encoding='utf-8') if file.exists() else records.render_log(record))
            (self.root/'runs/实验汇总.txt').write_text('\n\n'.join(texts),encoding='utf-8')
        if show:
            _show('<p>按保存时间查看最近10次实验。设置和结果见runs/实验汇总.txt，可直接复制给AI；对照结论应结合证据分析，准确率提高不作为必备条件。</p>'+_table(['实验','模型','结构','学习率','轮数','训练集','种子','基础验证','训练秒数'],[(x['name'],x['model'],x['structure'],x['lr'],x['epochs'],x['train_split'],x['seed'],f'{x["val_accuracy"]:.1%}',f'{x["seconds"]:.2f}') for x in rows[-10:]]))
        return rows

    def challenge(self):
        """单独评价原始父图、偏移、加粗、遮挡各640张；不训练、不混为一个分数。"""
        run=self._require(); rows=[]; previews=[]
        names={'original':'独立原始涂鸦（配对参照）','shift':'右移3、下移2像素','thick':'3×3邻域线条加粗','occlusion':'中央8×5像素遮挡'}
        for kind,name in names.items():
            split='challenge_'+kind; data=self.dataset(split)
            ev=core.evaluate_model(run.model,data); run.evaluations[split]=ev
            rows.append((name,len(data),f'{ev["accuracy"]:.1%}'))
            previews.append('<div class="dl-card">'+_image(data.x[0,0])+'<br>'+_esc(name)+'</div>')
        records.write_log(run.path,run.result,run.evaluations)
        self.compare(show=False)
        _show('<p><b>当前模型：'+_esc(run.name)+'</b>。以下是合成挑战，三种变换共用同一组独立原图；它们不是实际拍照。原图与变换图均未进入训练。原图是比较变换效果的参照，并不保证更难。</p><div class="dl-grid">'+''.join(previews)+'</div>'+_table(['类型','张数','准确率'],rows)+'<details><summary>展开：各类别挑战结果</summary>'+''.join('<p>'+_esc(names[k])+'</p>'+_table(['类别','准确率'],[(x['zh'],f'{x["accuracy"]:.1%}') for x in run.evaluations['challenge_'+k]['per_class']]) for k in names)+'</details>')
        return {split:{'accuracy':ev['accuracy'],'count':ev['count'],'per_class':ev['per_class']} for split,ev in run.evaluations.items() if split.startswith('challenge_')}

    def predict(self, image, **options):
        """image=路径/PIL/uint8数组；mode=drawing/photo，crop=(左,上,右,下)。

        返回预测dict。显示原图与模型所见28×28图，将路径、处理参数和前三项预测追加到TXT，不复制大图片。
        示例 lab.predict('my_images/doodle.png')；lab.predict('my_images/photo.jpg',mode='photo',crop=(20,20,400,400))。
        """
        run=self._require()
        if isinstance(image,(str,Path)):
            path=Path(image); image=path if path.is_absolute() else self.root/path
        pred=core.predict_image(run.model,image,**options)
        p=pred['prepared']
        source=str(image.relative_to(self.root)) if isinstance(image,Path) and image.is_relative_to(self.root) else str(image) if isinstance(image,Path) else 'PIL或数组输入'
        if not (run.path/'实验记录.txt').exists():records.write_log(run.path,run.result,run.evaluations)
        records.append_prediction(run.path,source,pred)
        _show('<p>使用模型：<b>'+_esc(run.name)+'</b><br>第一项预测：<b>'+_esc(pred['top'][0]['zh'])+'</b></p><div class="dl-grid"><div class="dl-card"><img style="max-width:250px;max-height:230px" src="'+_png(p['original'])+'"><br>输入原图</div><div class="dl-card"><img style="width:224px;image-rendering:pixelated" src="'+_png(p['processed'])+'"><br>模型实际看到的28×28图</div></div>'+_table(['猜测','输出分数'],[(x['zh'],f'{x["score"]:.3f}') for x in pred['top']])+'<p>分数不是保证正确的概率；类别外的图片也会被分到这八类之一。本次处理设置和预测已追加到当前实验的实验记录.txt；需要作为报告证据的图片请自行保留。</p>')
        return Prediction(pred)

    def experiment_panel(self):
        """用滑块与下拉设置实验；只有按训练按钮才生效。返回可供程序访问的控件字典。"""
        from .panels import experiment_panel
        return experiment_panel(self)

    def image_panel(self):
        """默认显示随包合成画板例图；可选择本地my_images图片并调整预处理。"""
        from .panels import image_panel
        return image_panel(self)

    def challenge_panel(self):
        """高级按钮，运行全部时不自动执行挑战评价。"""
        from .panels import challenge_panel
        return challenge_panel(self)

    def load(self, path):
        """实验目录str/Path → Experiment；验证权重后切换当前模型，重新评价基础验证。"""
        path=Path(path); path=path if path.is_absolute() else self.root/path
        model,result=core.load_model(path)
        ev=core.evaluate_model(model,self.dataset('val_basic'))
        self.latest=Experiment(result.get('experiment_name',path.name),model,result,{'val_basic':ev},path,0.0)
        _show('<p>✅ 已加载：'+_esc(self.latest.name)+'。后续图片预测使用该模型。</p>')
        return self.latest

    def export(self, experiments=None):
        """生成轻量提交材料，默认选第一次和当前实验；可传实验路径列表。

        保存两份TXT、模型、少量曲线和清除输出的程序，不复制课程数据或全部试跑。
        返回新建目录Path；Word报告需学生填写后提交；实验附件单独压缩。
        """
        run=self._require();model,_=core.load_model(run.path)
        repeat=core.evaluate_model(model,self.dataset('val_basic'))
        delta=float(np.max(np.abs(np.asarray(run.evaluations['val_basic']['scores'])-np.asarray(repeat['scores']))))
        if delta>1e-6:raise RuntimeError(f'重载预测不一致：{delta}')
        self.compare(show=False)
        target,sizes,count=records.submission(self,experiments)
        total=sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
        _show('<div class="dl-banner"><b>✅ 已准备轻量提交材料</b><br><code>'+_esc(target.relative_to(self.root))+'</code><br>'+str(count)+'次实验 · 当前目录约'+f'{total/1024:.0f}'+'KB（未填写报告）<br>当前模型保存加载分数差异：'+str(delta)+'</div><p>请填写目录内的报告_请填写.docx，重命名后作为主文件提交；实验附件文件夹单独压缩。实验附件中的实验汇总.txt可提供给AI辅助分析；课程数据和全部试跑不重复提交。</p>'+('<p>目前只有一次实验；完成有目的的对照后再导出。</p>' if count<2 else ''))
        return target


def start_lab(root='.'):
    """启动工作台，显示本地环境与固定标签；不联网、不自动训练。"""
    display(HTML('''<style>
    .doodle-lab{font-family:system-ui,sans-serif;line-height:1.7;color:#25334b;margin:10px 0;max-width:1000px}
    .doodle-lab .dl-banner{background:#ecf7f4;border-left:5px solid #24967c;padding:14px 18px;border-radius:8px}
    .doodle-lab .dl-grid{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}
    .doodle-lab .dl-card{background:#f5f7fc;border:1px solid #dfe5f1;padding:12px;border-radius:12px;text-align:center}
    .doodle-lab small{display:block;color:#697588}.doodle-lab table{border-collapse:collapse;width:100%;font-size:14px}
    .doodle-lab td,.doodle-lab th{padding:8px 12px;border-bottom:1px solid #e4e8ef;text-align:left;white-space:nowrap}
    .doodle-lab th{background:#edf2fa}.doodle-lab summary{cursor:pointer;padding:10px;background:#f4f6fa;border-radius:6px;margin:8px 0}
    .doodle-lab .dl-figure{max-width:100%;height:auto}
    </style>'''))
    lab=DoodleLab(root)
    import platform, torch
    _show('<div class="dl-banner"><b>✅ 涂鸦工作台已启动</b><br>数据来自本地 · 使用CPU · 无需CUDA<br>Python '+_esc(platform.python_version())+' / PyTorch '+_esc(torch.__version__)+'<br>识别类别：'+'、'.join(x['zh'] for x in lab.labels)+'</div>')
    return lab
