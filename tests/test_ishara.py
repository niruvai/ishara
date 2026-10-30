import sys, types, math, time, numpy as np
# ---- fake Quartz ----
Q = types.ModuleType("Quartz"); events=[]
for i,n in enumerate("kCGEventLeftMouseDown kCGEventLeftMouseUp kCGEventRightMouseDown kCGEventRightMouseUp kCGEventMouseMoved kCGEventLeftMouseDragged kCGEventRightMouseDragged".split()):
    setattr(Q,n,n)
for n in "kCGMouseButtonLeft kCGMouseButtonRight kCGHIDEventTap kCGMouseEventClickState kCGScrollEventUnitPixel".split(): setattr(Q,n,n)
for i,n in enumerate("kCGEventFlagMaskCommand kCGEventFlagMaskShift kCGEventFlagMaskAlternate kCGEventFlagMaskControl kCGEventFlagMaskSecondaryFn kCGEventFlagMaskNumericPad".split()): setattr(Q,n,1<<i)
class Ev(dict): pass
class P: 
    def __init__(s,x,y): s.x,s.y=x,y
class B:
    origin=P(0,0); size=types.SimpleNamespace(width=1440,height=900)
Q.CGDisplayBounds=lambda d:B(); Q.CGMainDisplayID=lambda:1
Q.CGEventCreate=lambda x:None; Q.CGEventGetLocation=lambda e:P(700,450)
Q.CGEventCreateMouseEvent=lambda n,t,pos,b:Ev(t=t,pos=pos)
Q.CGEventSetIntegerValueField=lambda e,f,v:e.__setitem__("clicks",v)
Q.CGEventPost=lambda tap,e:events.append(e)
Q.CGEventCreateScrollWheelEvent=lambda n,u,c,*a:Ev(t="scroll",v=a)
Q.CGEventCreateKeyboardEvent=lambda n,c,d:Ev(t="key",code=c,down=d)
Q.CGEventSetFlags=lambda e,f:e.__setitem__("flags",f)
sys.modules["Quartz"]=Q
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ishara
ishara.play=lambda *a:None
ishara.frontmost_app=lambda:"Finder"

# ---- synthetic hand ----
MCP={5:(-.025,-.085),9:(0,-.09),13:(.02,-.085),17:(.038,-.075)}
SEG={5:[.04,.025,.02],9:[.045,.028,.02],13:[.042,.027,.02],17:[.032,.02,.018]}
def hand(curls, thumb="open", ox=0.0, oy=0.0):
    W=np.zeros((21,3))
    for k,(mcp,c) in enumerate(zip((5,9,13,17),curls)):
        p=np.array([*MCP[mcp],0.0]); W[mcp]=p; th=0
        for j,L in enumerate(SEG[mcp]):
            th+=c; d=np.array([0,-math.cos(th),math.sin(th)]); p=p+L*d; W[mcp+1+j]=p
    W[1]=(-.02,-.02,0); W[2]=(-.04,-.04,0)
    if thumb=="open": W[3]=(-.06,-.055,0); W[4]=(-.08,-.065,0)
    elif thumb=="tuck": W[3]=(-.03,-.06,.02); W[4]=W[6]+np.array([.01,0,.01])
    elif thumb=="up": W[3]=(-.045,-.07,0); W[4]=(-.05,-.11,0)
    elif thumb=="right": W[3]=(.0,-.045,0); W[4]=(.04,-.05,0)
    img=np.zeros((21,2)); s=1.6  # meters->normalized
    img[:,0]=0.5+ox+W[:,0]*s/ (640/480); img[:,1]=0.62+oy+W[:,1]*s
    L=lambda a:[types.SimpleNamespace(x=v[0],y=v[1],z=v[2] if len(v)>2 else 0) for v in a]
    imgl=[types.SimpleNamespace(x=x,y=y,z=0) for x,y in img]
    return imgl, L(W)
def res(*hs):
    return types.SimpleNamespace(hand_landmarks=[h[0] for h in hs], hand_world_landmarks=[h[1] for h in hs],
        handedness=[[types.SimpleNamespace(category_name=n)] for n in ("Right","Left")][:len(hs)])

cfg=ishara.load_config()
m=ishara.Mouse(); d=ishara.CursorDriver(m); d.start()
tr=ishara.Tracker(cfg,m,d)
t=[100.0]
def step(r,n=1,dt=1/30):
    for _ in range(n):
        t[0]+=dt; tr.process(r,t[0],640/480); time.sleep(0.004)
def kinds(): return [e["t"] for e in events if e["t"]!="kCGEventMouseMoved" and e["t"]!="kCGEventLeftMouseDragged"]
OPEN=[0.05,0.05,0.05,0.05]
f=ishara.hand_features(np.array([[l.x*4/3,l.y] for l in hand(OPEN)[0]]), np.array([[l.x,l.y,l.z] for l in hand(OPEN)[1]]))
print("open reach",f["reach"].round(2),"thumb",round(f["thumb_ext"],2))
for name,c,th in [("dip",[0.45,.05,.05,.05],"open"),("fist",[1.5]*4,"tuck"),("thumbup",[1.5]*4,"up"),("scroll",[.05,1.5,1.5,.05],"open")]:
    f=ishara.hand_features(np.array([[l.x*4/3,l.y] for l in hand(c,th)[0]]), np.array([[l.x,l.y,l.z] for l in hand(c,th)[1]]))
    print(name,"idx_dip",round(f["index_dip"]),"pinky_dip",round(f["pinky_dip"]),"reach",f["reach"].round(2),"thumb",round(f["thumb_ext"],2),f["thumb_dir"])

step(res(hand(OPEN)),10); print("state after open:",tr.state)
step(res(hand(OPEN,ox=0.1)),15); time.sleep(0.2); print("cursor moved to",round(m.x),round(m.y))
events.clear()
step(res(hand([0.45,.05,.05,.05],ox=0.1)),4); step(res(hand(OPEN,ox=0.1)),4)
print("single click:",kinds())
events.clear()
step(res(hand([0.45,.05,.05,.05],ox=0.1)),4); step(res(hand(OPEN,ox=0.1)),4)
print("second quick click:",kinds(),[e.get("clicks") for e in events if "Down" in e["t"]])
t[0]+=1; events.clear()
step(res(hand([0.45,.05,.05,.05],ox=0.1)),3)
for i in range(15): step(res(hand([0.45,.05,.05,.05],ox=0.1+i*0.01)))
print("drag state:",tr.state, "dragged events:",sum(e["t"]=="kCGEventLeftMouseDragged" for e in events) )
step(res(hand(OPEN,ox=0.25)),4); print("after drag release:",kinds())
events.clear()
step(res(hand(OPEN)),15); step(res(hand([.05,.05,.05,.6])),4); step(res(hand(OPEN)),4)
print("right click:",kinds())
def scroll_sum():
    time.sleep(0.4); return sum(e["v"][0] for e in events if e["t"]=="scroll")
ROCK=[.05,1.5,1.5,.05]; PEACE=[.05,.05,1.5,1.5]
step(res(hand(OPEN)),15); events.clear()
step(res(hand(ROCK)),5); step(res(hand(ROCK)),15)
print("rock at center:",tr.state,"scroll sum",scroll_sum())
events.clear(); step(res(hand(ROCK,oy=-0.08)),30)
print("rock hand UP held 1s: sum",round(scroll_sum()),"(should be > 0, continuous)")
events.clear(); step(res(hand(OPEN)),15)
step(res(hand(PEACE)),5); events.clear(); step(res(hand(PEACE,oy=0.08)),30)
print("peace hand DOWN held 1s:",tr.state,"sum",round(scroll_sum()),"(should be < 0)")
step(res(hand(OPEN)),8); print("after open hand:",tr.state)
cfg["scroll_mode"]="drag"; step(res(hand(PEACE)),6); events.clear()
for i in range(10): step(res(hand(PEACE,oy=0.01*i)))
print("drag mode, hand moving down: sum",round(scroll_sum()),"(page follows hand)")
cfg["scroll_mode"]="joystick"; step(res(hand(OPEN)),8)
events.clear()
step(res(hand([1.5]*4,"tuck")),10); print("fist ->",tr.state)
step(res(hand([1.5]*4,"up")),20); print("thumb up keys:",[(e["code"],e["down"]) for e in events if e["t"]=="key"], tr.info)
# two-hand cross toggle
step(res(hand(OPEN)),5)
a=lambda x: hand(OPEN,ox=x)
for sx in (-0.2,0.2,-0.2):
    step(res(a(sx),a(-sx)),3)
print("after two-hand cross enabled =",tr.enabled, tr.state)
for sx in (0.2,-0.2,0.2):
    step(res(a(sx),a(-sx)),3)
print("cross again enabled =",tr.enabled)
d.running=False
