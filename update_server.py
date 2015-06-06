import sh
from toolz import thread_first, compose

def first(x):
    return x[0]

def get_matching_processes(pattern,m=sh):
    all_processes = m.ps('-A')
    target_processes = m.grep(all_processes,pattern, _ok_code = [0,1]).split('\n')[:-1]
    target_pids = map(compose(int,first,str.split,str),target_processes)
    return target_pids

def kill_pids(pids,m=sh):
    for pid in pids:
        m.kill(pid)
        
def kill_matching_processes(pattern,m=sh):
    pids = get_matching_processes(pattern,m)
    kill_pids(pids,m)        
    
def pull_latest_code(m=sh):
    m.git('fetch','--all')
    m.git('clean','-df')
    m.git('reset','--hard','origin/master')
    
def restart_ipython_server(m):
    m.nohup('ipython', 'notebook')

kill_matching_processes("ipython")
pull_latest_code()
restart_ipython_server()    
