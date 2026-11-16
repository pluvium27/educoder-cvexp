"""小组作业的中文文字记录和轻量提交材料；完整逐图结果只在内存中用于展示。"""
from pathlib import Path
from datetime import datetime
import json
import shutil
import torch

IMAGE_MARKER='\n附：图片尝试（探索记录，不是独立测试）\n'
CHALLENGE_NAMES={'val_basic':'基础验证','challenge_original':'挑战原图（配对参照）',
    'challenge_shift':'挑战：右移3、下移2像素','challenge_thick':'挑战：3×3邻域加粗',
    'challenge_occlusion':'挑战：中央8×5格遮挡'}


def compact_evaluation(ev):
    """完整评价dict → 总体、各类、混淆和至多8条错误；不保存逐图分数。"""
    result={k:ev[k] for k in ('accuracy','loss','count','per_class','confusion','dataset') if k in ev}
    if 'errors_preview' in ev:
        result['errors_preview']=ev['errors_preview']
    else:
        result['errors_preview']=[dict(row=i,true=ev['true_labels'][i],predicted=ev['predictions'][i],
            score=max(ev['scores'][i])) for i in ev.get('error_indices',[])[:8]]
    return result


def read_record(path):
    """实验目录 → 小配置与指标字典；同时兼容旧版experiment.json目录。"""
    path=Path(path)
    if (path/'experiment.json').exists():
        record=json.loads((path/'experiment.json').read_text(encoding='utf-8'))
        record['evaluations']={}
        for name in CHALLENGE_NAMES:
            file=path/(name+'.json')
            if file.exists():record['evaluations'][name]=compact_evaluation(json.loads(file.read_text(encoding='utf-8')))
        return record
    ckpt=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
    return ckpt['record']


def saved_runs(root):
    """按生成时间返回实验目录，只收录含模型与有效设置的目录。"""
    rows=[]
    for path in (Path(root)/'runs').glob('*/model.pt'):
        record=read_record(path.parent)
        rows.append((record.get('created_utc',''),path.parent,record))
    return [(p,r) for _,p,r in sorted(rows,key=lambda x:(x[0],x[1].name))]


def render_log(record,evaluations=None):
    c=record['config'];h=record['history'];last=h[-1];data=record['data'];env=record.get('environment',{})
    structure=c['custom_params'] if c['model']=='custom' else c['hidden_sizes'] if c['model']=='mlp' else c['channels']
    lines=['AI你画我猜 · 实验记录',f"实验名：{record.get('experiment_name','未命名')}",
        f"生成时间（UTC）：{record.get('created_utc','未记录')}",'',
        '一、训练设置',f"模型：{c['model']}；结构参数：{structure}；模型参数数量：{record['parameter_count']}",
        f"训练集：{c['train_split']}，{data['train']['count']}张；基础验证集：{data['validation']['count']}张",
        f"数据版本：{data['train']['data_version']}；类别："+'、'.join(f"{x['id']}={x['zh']}" for x in record['labels']),
        f"学习率：{c['lr']}；轮数：{c['epochs']}；每批图片数：{c['batch_size']}；随机种子：{c['seed']}",
        f"dropout：{c['dropout']}；weight_decay：{c['weight_decay']}；训练随机偏移：{c['augment_shift']}像素",
        f"设备：CPU；线程：{c['threads']}；Python：{env.get('python','未记录')}；PyTorch：{env.get('torch','未记录')}",
        f"系统：{env.get('platform','未记录')}；程序版本：{record.get('code_version','未记录')}",
        f"训练耗时：{record['seconds']:.2f}秒（不含完整保存/显示时间）",
        '每次从头训练，保存最后一轮模型；验证集公开供分析，结果不是独立测试成绩。',
        '', '二、逐轮结果（loss表示损失；准确率表示正确识别比例）',
        '轮次 | 训练loss | 验证loss | 训练准确率 | 验证准确率']
    lines += [f"{x['epoch']:>2} | {x['train_loss']:.4f} | {x['val_loss']:.4f} | {x['train_accuracy']:.2%} | {x['val_accuracy']:.2%}" for x in h]
    lines += ['',f"最后一轮：训练准确率{last['train_accuracy']:.2%}；基础验证准确率{last['val_accuracy']:.2%}"]
    summaries=dict(record.get('evaluations',{}));summaries.update(evaluations or {})
    for name,ev in summaries.items():
        lines += ['',CHALLENGE_NAMES.get(name,name),f"图片数：{ev['count']}；准确率：{ev['accuracy']:.2%}；loss：{ev['loss']:.4f}",
            '各类别：'+'；'.join(f"{x['zh']} {x['correct']}/{x['count']}（{x['accuracy']:.2%}）" for x in ev['per_class'])]
        if 'confusion' in ev:
            lines += ['混淆计数：行=真实类别，列=预测类别；顺序均为上方0—7。']
            lines += [' '.join(str(v) for v in row) for row in ev['confusion']]
        preview=compact_evaluation(ev).get('errors_preview',[])
        if preview:
            labels={x['id']:x['zh'] for x in record['labels']}
            lines += ['前8条错误摘要（按验证集顺序）：']
            lines += [f"验证行{x['row']}：标注{labels[x['true']]}，预测{labels[x['predicted']]}，最高输出分数{x['score']:.3f}" for x in preview]
    p=record['preprocess']
    lines += ['', '三、图片输入约定',
        f"28×28、黑底亮线、数值0—1；图案最长边默认占{p['fit_size']}格；找主体阈值{p['threshold']}；外部处理版本{p['version']}。",
        '模型只识别本课程八类；输出分数不是保证正确的概率。',
        '', '四、AI提问补充信息（供小组填写）',
        '研究问题：……；参照与改动：……；待核查判断：……。',
        '请先核对哪些条件不同，再解释证据；区分能支持的判断和还需验证的猜测。']
    return '\n'.join(lines)+'\n'


def write_log(path,record,evaluations=None):
    path=Path(path)/'实验记录.txt';tail=''
    if path.exists():
        old=path.read_text(encoding='utf-8')
        if IMAGE_MARKER in old:tail=IMAGE_MARKER+old.split(IMAGE_MARKER,1)[1]
    path.write_text(render_log(record,evaluations)+tail,encoding='utf-8')
    return path


def append_prediction(path,source,pred):
    file=Path(path)/'实验记录.txt';old=file.read_text(encoding='utf-8')
    p=pred['prepared']['info']
    line=(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 图片：{source}\n"
          f"处理设置：输入类型={p['mode']}，方向={p['polarity']}，裁剪={p.get('crop')}，找主体阈值={p['threshold']}，图案最长边={p['fit_size']}格\n"
          '前三项：'+'；'.join(f"{x['zh']}={x['score']:.3f}" for x in pred['top'])+'\n')
    file.write_text(old+('' if IMAGE_MARKER in old else IMAGE_MARKER)+line,encoding='utf-8')


def submission(lab,experiments=None):
    """生成提交材料目录，不复制数据或Notebook输出。

    默认只选第一次和当前实验；同一实验去重。可传实验路径列表选择其他证据。
    每次使用新目录，已填写的报告不会覆盖；报告需学生填写后提交，实验附件单独压缩。
    """
    records=saved_runs(lab.root)
    if experiments is None:
        chosen=[records[0][0],lab.latest.path] if records else [lab.latest.path]
    else:
        chosen=[Path(p) if Path(p).is_absolute() else lab.root/Path(p) for p in experiments]
    chosen=list(dict.fromkeys(p.resolve() for p in chosen))
    if not chosen:raise ValueError('请至少选择一个已完成的实验。')
    target=lab.root/'提交材料'/datetime.now().strftime('小组作业_%Y%m%d_%H%M%S_%f')
    target.mkdir(parents=True)
    shutil.copy2(lab.root/'报告模板.docx',target/'报告_请填写.docx')
    attachment=target/'实验附件';attachment.mkdir()
    # 提交程序保留一份；模型结构快照仅为自定义模型提供。
    shutil.copytree(lab.root/'资源/doodlelab',attachment/'资源/doodlelab',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    notebook=json.loads((lab.root/'AI你画我猜.ipynb').read_text(encoding='utf-8'))
    notebook.get('metadata',{}).pop('widgets',None)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
    (attachment/'AI你画我猜.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    for name in ['requirements-common.txt','LICENSE_CODE.txt']:shutil.copy2(lab.root/'资源'/name,attachment/'资源'/name)
    texts=[];sizes=[]
    for number,path in enumerate(chosen,1):
        record=read_record(path)
        dest=attachment/'实验'/f"{number:02d}_{path.name}";dest.mkdir(parents=True)
        # 旧记录无需迁移或删除；导出时转换为相同的小记录格式。
        checkpoint=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
        clean=json.loads(json.dumps(record,ensure_ascii=False))
        evaluations={k:compact_evaluation(v) for k,v in clean.get('evaluations',{}).items()}
        if not evaluations and (path/'val_basic.json').exists():
            evaluations['val_basic']=compact_evaluation(json.loads((path/'val_basic.json').read_text(encoding='utf-8')))
        clean['evaluations']=evaluations;clean.pop('model_sha256',None)
        checkpoint['record']=clean;torch.save(checkpoint,dest/'model.pt')
        if (path/'实验记录.txt').exists():shutil.copy2(path/'实验记录.txt',dest/'实验记录.txt')
        else:
            for kind in CHALLENGE_NAMES:
                file=path/(kind+'.json')
                if file.exists():evaluations[kind]=compact_evaluation(json.loads(file.read_text(encoding='utf-8')))
            write_log(dest,record,evaluations)
        curve=path/'训练曲线.png'
        if not curve.exists():curve=path/'curves.png'
        if curve.exists():shutil.copy2(curve,dest/'训练曲线.png')
        custom=path/'source/doodlelab/custom_model.py'
        if record['config']['model']=='custom' and custom.exists():
            out=dest/'source/doodlelab';out.mkdir(parents=True);shutil.copy2(custom,out/custom.name)
        texts += [(dest/'实验记录.txt').read_text(encoding='utf-8')]
        sizes += [sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())]
    (attachment/'实验汇总.txt').write_text('\n\n'+'\n\n'.join('【实验'+str(i+1)+'】\n'+t for i,t in enumerate(texts)),encoding='utf-8')
    (target/'提交前请读.txt').write_text(
        '提交材料整理要求\n'
        '1. 使用Word填写报告_请填写.docx，将提示替换为本组真实实验与分析，重命名为小组编号_你画我猜报告.docx。报告为主要评分材料。\n'
        '2. 报告正文选择少量关键图表，包含AI核查、小组讨论与个人反思。无需另交完整聊天记录。\n'
        f'3. 实验附件含{len(chosen)}次实际实验；必做任务应包含参照与改动两次实验。未完成对照时，补做后重新导出。\n'
        '4. 将实验附件文件夹压缩为小组编号_实验附件.zip，与报告一并提交。无需重复提交课程数据、Python环境、全部试跑或原始大照片。\n'
        '5. 提交前核对报告与实验附件一致；若平台只接收一个文件，将报告和附件合并压缩，报告放在压缩包顶层。\n'
        '复运行：完整解压课程包，将实验附件中的Notebook、资源及实验目录复制到课程目录，合并资源文件夹并保留课程数据。\n'
        '启动工作台后使用lab.load("实验/相应目录")查看结果或预测。自定义模型需先恢复相应源码快照并重启内核。\n',encoding='utf-8')
    return target,sizes,len(chosen)
