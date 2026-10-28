"""新增预测的单一完成回执；半成品不视为完成。"""
import os
import uuid
from p4_core import ROOT,STAGE,write,read,file_hash,storage
from tlcls.p2_core import fsync_directory


def completed(directory,identity):
    path=directory/'completed.json'
    if not path.exists():return None
    value=read(path);assert value['identity']==identity and value['status']=='completed'
    data=(ROOT/value['data_path']).resolve();assert data.is_relative_to(directory.resolve())
    assert file_hash(data)==value['data_sha256']
    return value


def publish(directory,identity,writer,metadata,fault=lambda _:None):
    storage();directory.mkdir(parents=True,exist_ok=True)
    target=directory/(uuid.uuid4().hex+'.npz');temporary=target.with_suffix('.npz.tmp')
    with temporary.open('xb') as file:
        writer(file);file.flush();os.fsync(file.fileno())
    fault('file_synced');digest=file_hash(temporary);os.replace(temporary,target);fsync_directory(directory)
    fault('data_published')
    value={'status':'completed','identity':identity,'data_path':target.relative_to(ROOT).as_posix(),'data_sha256':digest,**metadata}
    write(directory/'completed.json',value);fault('receipt_published')
    return value
