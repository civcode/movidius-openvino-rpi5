"""OpenCV display helpers that never block inference."""
import subprocess, threading, time
try:
    import cv2
except ImportError:
    cv2=None

class WindowWatcher:
    def __init__(self,title):
        self._title=title; self._last_poll=0.0; self._closed=False
    def closed(self):
        now=time.monotonic()
        if now-self._last_poll<0.5: return self._closed
        self._last_poll=now
        try:
            out=subprocess.run(["xwininfo","-root","-tree"],capture_output=True,text=True,timeout=2).stdout
            self._closed=self._title not in out
        except (OSError,subprocess.SubprocessError):
            pass
        return self._closed

class DisplayPump:
    def __init__(self,title,draw,size_spec=None,idle_sleep=0.001):
        self._title=title; self._draw=draw; self._size_spec=size_spec; self._idle=idle_sleep
        self._cv=threading.Condition(); self._item=None; self._pending=False; self._stop=False
        self._closed=False; self._sized=False; self._seen_visible=False
        self._thread=threading.Thread(target=self._run,daemon=True); self._thread.start()
    def post(self,item):
        with self._cv: self._item=item; self._pending=True; self._cv.notify()
    def closed(self): return self._closed or self._stop
    def _run(self):
        if cv2 is None: return
        cv2.namedWindow(self._title,cv2.WINDOW_NORMAL)
        while not self._stop:
            with self._cv:
                while not self._pending and not self._stop: self._cv.wait(0.05)
                item,self._pending=self._item,False
            if self._stop: break
            if item is None: continue
            try:
                img=self._draw(item)
                if img is not None:
                    cv2.imshow(self._title,img)
                    if not self._sized:
                        if self._size_spec is not None: cv2.resizeWindow(self._title,*self._size_spec)
                        else: cv2.resizeWindow(self._title,img.shape[1],img.shape[0])
                        self._sized=True
                key=cv2.waitKey(1)&0xff
                if key==ord("q") or key==27: self._closed=True; continue
                try: vis=cv2.getWindowProperty(self._title,cv2.WND_PROP_VISIBLE)
                except Exception: vis=-1
                if vis is not None and vis>=1: self._seen_visible=True
                elif self._seen_visible: self._closed=True
            except Exception:
                self._closed=True; break
    def close(self):
        self._stop=True
        with self._cv: self._cv.notify_all()
        self._thread.join(timeout=2.0)
        if cv2 is not None:
            try: cv2.destroyWindow(self._title)
            except Exception: pass
