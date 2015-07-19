from toolz import thread_first, thread_last, curry, groupby, pipe
import os

config = dict(c1 = "TxRed",
              c2 = "FITC",
              c3 = "DAPI")

directory = '/Users/jonah/Desktop/SRF Internship/APB ssC optimization data/Raw'
files = ['60x APB +RNase A 143b DAPI 01.tif', '60x APB +RNase A 143b DAPI 02.tif', '60x APB +RNase A 143b FITC 01.tif', '60x APB +RNase A 143b FITC 02.tif', '60x APB +RNase A 143b TxRed 01.tif', '60x APB +RNase A 143b TxRed 02.tif', '60x APB +RNase A U2OS DAPI 01.tif', '60x APB +RNase A U2OS DAPI 02.tif', '60x APB +RNase A U2OS FITC 01.tif', '60x APB +RNase A U2OS FITC 02.tif', '60x APB +RNase A U2OS TxRed 01.tif', '60x APB +RNase A U2OS TxRed 02.tif', '60x IF only 143b DAPI 01.tif', '60x IF only 143b DAPI 02.tif', '60x IF only 143b FITC 01.tif', '60x IF only 143b FITC 02.tif', '60x IF only 143b TxRed 01.tif', '60x IF only 143b TxRed 02.tif', '60x IF only U2OS DAPI 01.tif', '60x IF only U2OS DAPI 02.tif', '60x IF only U2OS FITC 01.tif', '60x IF only U2OS FITC 02.tif', '60x IF only U2OS TxRed 01.tif', '60x IF only U2OS TxRed 02.tif', '60x IgG APB probe after denature 143b DAPI 01.tif', '60x IgG APB probe after denature 143b DAPI 02.tif', '60x IgG APB probe after denature 143b FITC 01.tif', '60x IgG APB probe after denature 143b FITC 02.tif', '60x IgG APB probe after denature 143b TxRed 01.tif', '60x IgG APB probe after denature 143b TxRed 02.tif', '60x IgG APB probe after denature U2OS DAPI 01.tif', '60x IgG APB probe after denature U2OS DAPI 02.tif', '60x IgG APB probe after denature U2OS FITC 01.tif', '60x IgG APB probe after denature U2OS FITC 02.tif', '60x IgG APB probe after denature U2OS TxRed 01.tif', '60x IgG APB probe after denature U2OS TxRed 02.tif', '60x IgG APB probe before denature 143b DAPI 01.tif', '60x IgG APB probe before denature 143b DAPI 02.tif', '60x IgG APB probe before denature 143b FITC 01.tif', '60x IgG APB probe before denature 143b FITC 02.tif', '60x IgG APB probe before denature 143b TxRed 01.tif', '60x IgG APB probe before denature 143b TxRed 02.tif', '60x IgG APB probe before denature U2OS DAPI 01.tif', '60x IgG APB probe before denature U2OS DAPI 02.tif', '60x IgG APB probe before denature U2OS FITC 01.tif', '60x IgG APB probe before denature U2OS FITC 02.tif', '60x IgG APB probe before denature U2OS TxRed 01.tif', '60x IgG APB probe before denature U2OS TxRed 02.tif', '60x IgG ssC 143b DAPI 01.tif', '60x IgG ssC 143b DAPI 02.tif', '60x IgG ssC 143b FITC 01.tif', '60x IgG ssC 143b FITC 02.tif', '60x IgG ssC 143b TxRed 01.tif', '60x IgG ssC 143b TxRed 02.tif', '60x IgG ssC U2OS DAPI 01.tif', '60x IgG ssC U2OS DAPI 02.tif', '60x IgG ssC U2OS FITC 01.tif', '60x IgG ssC U2OS FITC 02.tif', '60x IgG ssC U2OS TxRed 01.tif', '60x IgG ssC U2OS TxRed 02.tif', '60x ssC +RNase A 143b DAPI 01.tif', '60x ssC +RNase A 143b DAPI 02.tif', '60x ssC +RNase A 143b FITC 01.tif', '60x ssC +RNase A 143b FITC 02.tif', '60x ssC +RNase A 143b TxRed 01.tif', '60x ssC +RNase A 143b TxRed 02.tif', '60x ssC +RNase A U2OS DAPI 01.tif', '60x ssC +RNase A U2OS DAPI 02.tif', '60x ssC +RNase A U2OS FITC 01.tif', '60x ssC +RNase A U2OS FITC 02.tif', '60x ssC +RNase A U2OS TxRed 01.tif', '60x ssC +RNase A U2OS TxRed 02.tif', '60x ssC -probe 143b DAPI 01.tif', '60x ssC -probe 143b DAPI 02.tif', '60x ssC -probe 143b FITC 01.tif', '60x ssC -probe 143b FITC 02.tif', '60x ssC -probe 143b TxRed 01.tif', '60x ssC -probe 143b TxRed 02.tif', '60x ssC -probe U2OS DAPI 01.tif', '60x ssC -probe U2OS DAPI 02.tif', '60x ssC -probe U2OS FITC 01.tif', '60x ssC -probe U2OS FITC 02.tif', '60x ssC -probe U2OS TxRed 01.tif', '60x ssC -probe U2OS TxRed 02.tif', '60x ssC -RNase A 143b DAPI 01.tif', '60x ssC -RNase A 143b DAPI 02.tif', '60x ssC -RNase A 143b FITC 01.tif', '60x ssC -RNase A 143b FITC 02.tif', '60x ssC -RNase A 143b TxRed 01.tif', '60x ssC -RNase A 143b TxRed 02.tif', '60x ssC -RNase A U2OS DAPI 01.tif', '60x ssC -RNase A U2OS DAPI 02.tif', '60x ssC -RNase A U2OS FITC 01.tif', '60x ssC -RNase A U2OS FITC 02.tif', '60x ssC -RNase A U2OS TxRed 01.tif', '60x ssC -RNase A U2OS TxRed 02.tif']

# {a:b} -> [string] -> {a:b}
def filter_keys(d,keys):
    return dict(filter(lambda x: x[0] in keys,d.items()))

# {c1:string, c2:string, c3:string} -> string -> {group:string, channel:string, filename:string}
@curry
def parse(config,filename):
    for channel, pattern in config.iteritems():
        if pattern in filename:
            return dict(group = filename.replace(pattern,''),
                        channel =  channel,
                        filename = filename)

# [{group:string, channel:string, filename:string}] -> {c1:string, c2:string, c3:string}
def make_channel_dict(items):
    return {d['channel']:d['filename'] for d in items}

# (b -> c) -> {a:b} -> {a:c}
def mapvals(f,d):
    return {key:f(value) for key,value in d.iteritems()}

# {a -> b -> c} -> {a:b} -> [c]
def mapdict(f,d):
    return [f(key,value) for key,value in d.iteritems()]

# string -> [string] -> {c1:string, c2:string, c3:string} -> string
def generate_macro(filepath,filenames,config):
    return thread_last(filenames,
                      (map,parse(config)),
                      (groupby,lambda d: d['group']),
                      (mapvals,make_channel_dict),
                      (mapdict,create_task(filepath)),
                      (lambda x: '\n'.join(x)))

# string -> string -> {c1:string, c2:string, c3:string} -> string
@curry
def create_task(filepath,key,val):
    save_path = os.path.split(filepath)[0]
    return pipe([make_mkdir_cmd(os.path.join(save_path,key.rstrip('.tif'))),
                 make_open_cmd(os.path.join(filepath,val['c1'])),
                 make_open_cmd(os.path.join(filepath,val['c2'])),
                 make_open_cmd(os.path.join(filepath,val['c3'])),
                 make_merge_cmd(val),
                 make_save_cmd(os.path.join(save_path,key.rstrip('.tif'),'merge.jpg')),
                 'close("merge.jpg");',
                 make_merge_cmd(filter_keys(val,['c1'])),
                 make_save_cmd(os.path.join(save_path,key.rstrip('.tif'),'red.jpg')),
                 'close("red.jpg");',
                 make_merge_cmd(filter_keys(val,['c2'])),
                 make_save_cmd(os.path.join(save_path,key.rstrip('.tif'),'green.jpg')),
                 'close("green.jpg");',
                 make_merge_cmd(filter_keys(val,['c3'])),
                 make_save_cmd(os.path.join(save_path,key.rstrip('.tif'),'blue.jpg')),
                 'close("blue.jpg");',
                 'run("Close All");'],
                lambda x: '\n'.join(x))

# string -> string
def make_open_cmd(f):
    return 'open("{}");'.format(f)

# {a:b} -> string
def make_merge_cmd(d):
    return 'run("Merge Channels...","{} keep");'.format(dict_to_string(d))

def dict_to_string(d):
    return ' '.join(["{}=[{}]".format(key,val) for key,val in d.iteritems()])

def make_save_cmd(f):
    return 'saveAs("Jpeg", "{}");'.format(f)

# string -> string
def make_mkdir_cmd(filepath):
    return 'File.makeDirectory("{}");'.format(filepath)

def drop_path_dir(path):
    return path.split('/')

print generate_macro(directory,files,config)