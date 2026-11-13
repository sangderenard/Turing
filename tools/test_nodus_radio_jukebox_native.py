"""Silent gated native track proof. Args: radio.exe scope.dll fft.dll router.exe.
Uses disposable PCM and native loopback router, never a service or real device.
"""
from pathlib import Path
import os
from unittest.mock import patch
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from meowdoku.nodus_radio_jukebox_runner import NativeTrack, PcmTap
from meowdoku.nodus_radio_playback import Playback

def port(kind):
    with socket.socket(socket.AF_INET,kind) as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]

def main():
    radio,scope,fft,router=sys.argv[1:5]
    sink=Playback(sys.argv[5]) if len(sys.argv)>5 else None
    config=dict(executable=radio,scope_dll=scope,fft_dll=fft,router_port=port(socket.SOCK_STREAM),pcm_port=port(socket.SOCK_DGRAM),spectrum_port=port(socket.SOCK_DGRAM))
    with tempfile.TemporaryDirectory(prefix='radio-gated-') as td:
        wav=Path(td)/'track.wav'
        with wave.open(str(wav),'wb') as f:
            f.setparams((2,2,48000,0,'NONE','not compressed'));f.writeframes(b'\1\0\2\0'*(48000+7))
        proc=subprocess.Popen([router,'127.0.0.1',str(config['router_port'])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        track=NativeTrack(config,{'path':str(wav)})
        tap=PcmTap();sink_stop=threading.Event();sink_thread=None;written=[];errors=[]
        try:
            if sink:
                with patch.dict(os.environ,SDL_AUDIODRIVER='dummy'):sink.start()
                def output():
                    try:
                        while not sink_stop.is_set() or not tap.blocks.empty():
                            block=tap.take(.01)
                            if block is not None:sink.write(block);written.append(len(block)//4)
                    except Exception as exc:errors.append(exc)
                sink_thread=threading.Thread(target=output);sink_thread.start()
                track.on_pcm=tap.put
            deadline=time.monotonic()+5
            while True:
                try:
                    with socket.create_connection(('127.0.0.1',config['router_port']),timeout=.1):break
                except OSError:
                    assert time.monotonic()<deadline;time.sleep(.05)
            cancel=threading.Event(); timer=threading.Timer(15,cancel.set); timer.start()
            try: track.run(cancel)
            finally: timer.cancel()
            assert not cancel.is_set(), "native track timeout"
            state=track.sub.buffer.status()
            assert state['ended'] and state['received']==188,state
            assert not any(state[k] for k in ('lost','late','rejected','resyncs')),state
            assert track.spectrum_peer is not None
            if sink:
                track.close()
                track=NativeTrack(config,{'path':str(wav)});track.on_pcm=tap.put
                timer=threading.Timer(15,cancel.set);timer.start()
                try:track.run(cancel)
                finally:timer.cancel()
                assert not cancel.is_set()
                sink_stop.set();sink_thread.join(3)
                assert not sink_thread.is_alive() and not errors,errors
                sink.finish()
                assert sum(written)==96014,sum(written)
                print('PASS same native PA sink followed two real sender/SSRC changes: 96014 frames including both tails; SDL dummy')
            print('PASS gated native source: 48007 frames, 188 packets, full short tail, verified EOF; zero loss/reject/resync; silent SDL dummy')
        finally:
            sink_stop.set()
            if sink_thread:sink_thread.join(3)
            if sink:sink.stop()
            tap.close();track.close();proc.terminate();proc.wait(timeout=5)
if __name__=='__main__':main()
