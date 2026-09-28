"""Camera and synthetic-source helpers shared by all streaming examples."""
import sys, threading, time
try:
    import cv2
except ImportError:
    cv2=None

def _note(msg):
    print(msg,file=sys.stderr,flush=True)

def add_camera_capture_args(ap,width_opt="--camera-width",height_opt="--camera-height",
                            width_default=0,height_default=0,fourcc_default="MJPG"):
    width_opt=[width_opt] if isinstance(width_opt,str) else list(width_opt)
    height_opt=[height_opt] if isinstance(height_opt,str) else list(height_opt)
    ap.add_argument(*width_opt,type=int,default=width_default,help="request capture width (0 = device default)")
    ap.add_argument(*height_opt,type=int,default=height_default,help="request capture height (0 = device default)")
    ap.add_argument("--camera-fps",type=float,default=0.0,help="request capture frame rate (0 = device default)")
    ap.add_argument("--camera-fourcc",default=fourcc_default,help="request pixel format (MJPG / YUYV / none)")

def fourcc_from_tag(tag):
    tag=(tag or "").strip().upper()
    if tag in ("","NONE","DEFAULT","0"): return None
    if len(tag)!=4: raise ValueError("--camera-fourcc needs a 4-char tag like MJPG or YUYV")
    if cv2 is None: raise RuntimeError("OpenCV is required for --camera-fourcc")
    return cv2.VideoWriter_fourcc(*tag)

def fourcc_tag(value):
    if not value: return "none"
    tag="".join(chr((int(value)>>(8*i))&0xff) for i in range(4))
    return tag.strip() or "none"

def configure_camera(cap,fourcc=None,width=0,height=0,fps=0.0):
    if fourcc is not None: cap.set(cv2.CAP_PROP_FOURCC,fourcc)
    if width: cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    if fps and fps>0: cap.set(cv2.CAP_PROP_FPS,fps)
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fourcc_tag(cap.get(cv2.CAP_PROP_FOURCC)),cap.get(cv2.CAP_PROP_FPS))

def open_camera(index,fourcc=None,width=0,height=0,fps=0.0):
    if cv2 is None: raise RuntimeError("OpenCV is required")
    cap=cv2.VideoCapture(index,cv2.CAP_V4L2)
    if not cap.isOpened(): return None,None
    accepted=configure_camera(cap,fourcc,width,height,fps)
    if fourcc is not None and accepted[2]!=fourcc_tag(fourcc):
        _note("camera %d: no %s mode, driver kept %s - renegotiating without a format request" %
              (index,fourcc_tag(fourcc),accepted[2]))
        cap.release(); cap=cv2.VideoCapture(index,cv2.CAP_V4L2)
        if not cap.isOpened(): return None,None
        accepted=configure_camera(cap,None,width,height,fps)
    return cap,accepted

def note_camera_settings(index,accepted,requested_fourcc=None,requested_fps=0.0,requested_size=None):
    w,h,tag,fps=accepted
    _note("camera %d: %dx%d fourcc=%s nominal=%.0f fps (requested fourcc=%s fps=%.0f%s)" %
          (index,w,h,tag,fps,fourcc_tag(requested_fourcc) if requested_fourcc is not None else "none",
           requested_fps or 0.0,"" if not requested_size else " size=%dx%d"%requested_size))

class LatestFrame:
    def __init__(self,cap,history=32):
        self._cap=cap; self._cv=threading.Condition(); self._frame=None; self._eof=False; self._stop=False
        self._arrivals=[]; self._captures=0; self._reads=0; self._history=max(2,history); self._last_wait=0.0
        self._thread=threading.Thread(target=self._drain,daemon=True); self._thread.start()
    def _drain(self):
        while not self._stop:
            ok,frame=self._cap.read()
            if not ok:
                with self._cv: self._eof=True; self._cv.notify_all()
                return
            now=time.monotonic()
            with self._cv:
                self._frame=frame; self._captures+=1; self._arrivals.append(now)
                if len(self._arrivals)>self._history: del self._arrivals[:-self._history]
                self._cv.notify_all()
    def read(self):
        t0=time.monotonic()
        with self._cv:
            while self._frame is None and not self._eof: self._cv.wait()
            if self._frame is None: return None
            frame,self._frame=self._frame,None; self._reads+=1; self._last_wait=time.monotonic()-t0; return frame
    @property
    def last_wait_s(self): return self._last_wait
    def dropped(self):
        with self._cv: return max(0,self._captures-self._reads)
    def camera_fps(self,n=16):
        with self._cv: stamps=list(self._arrivals)
        stamps=stamps[-(n+1):]
        if len(stamps)<2: return 0.0
        span=stamps[-1]-stamps[0]; return (len(stamps)-1)/span if span>0 else 0.0
    def close(self):
        self._stop=True; self._thread.join(timeout=2.0)
        try: self._cap.release()
        except Exception: pass

def load_static_image(path):
    if cv2 is None: raise RuntimeError("OpenCV is required for --fake-camera")
    img=cv2.imread(path)
    if img is None: raise RuntimeError("cannot read fake-camera image %s"%path)
    return img

def add_fake_camera_args(ap,default_image=None):
    ap.add_argument("--fake-camera",action="store_true",help="serve frames from a static image instead of webcam")
    ap.add_argument("--fake-camera-fps",type=float,default=30.0,help="fake frame deliveries per second (0 = unbounded)")
    ap.add_argument("--fake-camera-image",default=default_image,help="static image to serve")

class FakeCamera:
    def __init__(self,image_bgr,fps=30.0):
        self.image=image_bgr; self.dt=(1.0/fps) if fps and fps>0 else 0.0
        self._prev=time.monotonic(); self._delivered=0; self._open=True
    def isOpened(self): return self._open
    def read(self):
        self._delivered+=1
        if self.dt:
            due=self._prev+self.dt; now=time.monotonic()
            if due>now: time.sleep(due-now)
            self._prev=time.monotonic()
        return True,self.image
    def get(self,prop):
        if cv2 is None: return 0
        h,w=self.image.shape[:2]
        return {cv2.CAP_PROP_FRAME_WIDTH:w,cv2.CAP_PROP_FRAME_HEIGHT:h,
                cv2.CAP_PROP_FPS:(1.0/self.dt) if self.dt else 0.0}.get(prop,0)
    def set(self,prop,value): return False
    def release(self): self._open=False
