"""Direct Colab process output with genuine child tqdm and elapsed-time heartbeat."""
from __future__ import annotations
import os,sys,pty,time,select,signal,codecs,subprocess
from pathlib import Path


def run_live(cmd,cwd,description,logfile=None):
    env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8','TERM':'xterm','COLUMNS':'140'}
    master,slave=pty.openpty();proc=None
    started=time.monotonic();last_output=started;last_beat=started
    phase='Initializing';tail=''
    print('='*80,'[START] '+description,'='*80,sep='\n',flush=True)
    try:
        proc=subprocess.Popen([str(x) for x in cmd],cwd=str(cwd),env=env,stdin=subprocess.DEVNULL,
                              stdout=slave,stderr=slave,start_new_session=True,close_fds=True)
        os.close(slave);slave=None
        decoder=codecs.getincrementaldecoder('utf-8')('replace')
        if logfile:Path(logfile).parent.mkdir(parents=True,exist_ok=True)
        f=open(logfile,'a',encoding='utf-8') if logfile else None
        try:
            while True:
                r,_,_=select.select([master],[],[],1.0)
                if r:
                    try:raw=os.read(master,32768)
                    except OSError:raw=b''
                    if raw:
                        chunk=decoder.decode(raw)
                        sys.stdout.write(chunk);sys.stdout.flush()
                        if f:f.write(chunk);f.flush()
                        last_output=time.monotonic();tail=(tail+chunk)[-1300:]
                        if '[STAGE]' in chunk:phase=chunk.split('[STAGE]')[-1].split('\n')[0].strip()
                    elif proc.poll() is not None:break
                elif proc.poll() is not None:break
                now=time.monotonic()
                if now-last_beat>=30:
                    print(f'\n[ALIVE] {description} | stage={phase} | elapsed={(now-started)/60:.1f}min | quiet={(now-last_output)/60:.1f}min',flush=True)
                    if now-last_output>600:print('[WARNING] No child output for >10 minutes; investigate dataset index, GPU and mounted files.',flush=True)
                    last_beat=now
            if proc.wait()!=0:raise subprocess.CalledProcessError(proc.returncode,cmd,output=tail)
        finally:
            if f:f.close()
    except BaseException:
        if proc and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=8)
            except Exception:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        raise
    finally:
        if slave is not None:os.close(slave)
        os.close(master)
    print('[SUCCESS]',description,flush=True)
