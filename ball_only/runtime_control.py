"""GPU duty-cycle feedback and thermal monitoring without changing model quality.

The 90--95% goal is an averaged utilization target, not an instantaneous cap.
Short inference bursts can still report 100%. CUDA/OOM/model errors propagate.
"""
from contextlib import contextmanager
from dataclasses import asdict,dataclass
import csv,errno,math,os,re,subprocess,time,threading


class ThermalLimitError(RuntimeError):
    """Stop this work unit so the supervisor can preserve/retry its checkpoint."""


class TelemetryUnavailableError(RuntimeError):
    """Repeated loss of the requested GPU safety/utilization monitoring."""


class RuntimeCancelledError(RuntimeError):
    pass


def _number(value):
    try:
        result=float(str(value).strip())
        return result if math.isfinite(result) else None
    except (ValueError,TypeError):
        return None


@dataclass
class GpuTelemetry:
    gpu_index: int
    uuid: str
    utilization_percent: float|None
    memory_used_mib: float|None
    memory_total_mib: float|None
    temperature_c: float|None
    power_draw_w: float|None
    power_limit_w: float|None
    sampled_at_unix: float


class NvidiaSmiTelemetry:
    def __init__(self,gpu_index=0,timeout_s=3.):
        self.gpu_index=int(gpu_index);self.timeout_s=float(timeout_s)
        self.hardware_limits_c={};self._limits_checked=False

    def _run(self,args):
        kwargs=dict(capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=self.timeout_s,check=True)
        if os.name=='nt':kwargs['creationflags']=getattr(subprocess,'CREATE_NO_WINDOW',0)
        return subprocess.run(['nvidia-smi','-i',str(self.gpu_index),*args],**kwargs).stdout

    def read(self):
        fields='index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit'
        output=self._run(['--query-gpu='+fields,'--format=csv,noheader,nounits'])
        rows=list(csv.reader(output.strip().splitlines()))
        if len(rows)!=1 or len(rows[0])!=8:raise ValueError('Unexpected nvidia-smi telemetry format')
        values=[s.strip() for s in rows[0]]
        sample=GpuTelemetry(int(values[0]),values[1],*[_number(v) for v in values[2:]],time.time())
        if sample.utilization_percent is None or sample.temperature_c is None:raise ValueError('GPU utilization/temperature telemetry unavailable')
        if not self._limits_checked:
            self._limits_checked=True
            try:
                text=self._run(['-q','-d','TEMPERATURE'])
                # Only absolute limits. Newer drivers report relative T.Limit
                # margins; do not mistake those for absolute temperatures.
                for key,label in [('shutdown','GPU Shutdown Temp'),('slowdown','GPU Slowdown Temp'),('max_operating','GPU Max Operating Temp')]:
                    match=re.search(r'^\s*'+re.escape(label)+r'\s*:\s*(\d+(?:\.\d+)?)\s*C\s*$',text,re.M)
                    if match:self.hardware_limits_c[key]=float(match.group(1))
            except (OSError,ValueError,subprocess.SubprocessError):
                pass
        return sample


class RuntimeController:
    """Call ``with controller.gpu_work(): ...`` around each inference call.

    Alternatively call ``after_gpu_work(seconds)`` after caller-synchronized
    GPU work. ``wrap_forward(model)`` wraps each SFR model forward. Serialize GPU
    phases. Set ``SNP_GPU_LOCK_FILE`` (or ``process_lock_file``) to the same path
    in cooperating processes to serialize GPU work and its required idle time.
    Nested contexts on this controller are reentrant.
    """
    def __init__(self,*,gpu_index=0,target_low=90.,target_high=95.,
                 telemetry_interval_s=5.,soft_temperature_c=78.,
                 pause_temperature_c=82.,critical_temperature_c=85.,
                 max_cooling_seconds=120.,max_telemetry_failures=3,
                 telemetry=None,cuda_sync=None,clock=None,sleep=None,stop_event=None,
                 process_lock_file=None):
        if not 0<target_low<target_high<100:raise ValueError('Utilization targets must satisfy 0 < low < high < 100')
        if not soft_temperature_c<pause_temperature_c<critical_temperature_c:raise ValueError('Require soft < pause < critical temperatures')
        self.target_low=float(target_low);self.target_high=float(target_high)
        self.target=(self.target_low+self.target_high)/2
        self.duty=self.target/100.;self.effective_duty=self.duty
        self.telemetry_interval_s=float(telemetry_interval_s)
        self.soft_temperature_c=float(soft_temperature_c);self.pause_temperature_c=float(pause_temperature_c);self.critical_temperature_c=float(critical_temperature_c)
        self.max_cooling_seconds=float(max_cooling_seconds);self.max_telemetry_failures=int(max_telemetry_failures)
        self.reader=telemetry or NvidiaSmiTelemetry(gpu_index)
        self._sync_fn=cuda_sync;self._clock=clock or time.perf_counter;self._sleep_fn=sleep or time.sleep;self.stop_event=stop_event
        self.last_sample=None;self.last_error=None;self.telemetry_failures=0;self._sampled_monotonic=None;self._last_return=None
        self.total_gpu_work_seconds=0.;self.total_throttle_seconds=0.;self.work_calls=0
        self.state='starting';self._inside=False;self._frame_start=None
        self._lock=threading.RLock();self._utilization_samples=[]
        self._critical_reason=None
        lock_file=os.environ.get('SNP_GPU_LOCK_FILE') if process_lock_file is None else process_lock_file
        self.process_lock_file=os.path.abspath(os.fspath(lock_file)) if lock_file else None
        if self.process_lock_file:os.makedirs(os.path.dirname(self.process_lock_file),exist_ok=True)

    @contextmanager
    def _process_gate(self):
        if self.process_lock_file is None:
            yield
            return
        # OS-owned locks are released even if a worker exits or is killed. Keep
        # the handle open throughout inference, synchronization and throttle.
        with open(self.process_lock_file,'a+b',buffering=0) as handle:
            if os.fstat(handle.fileno()).st_size==0:handle.write(b'\0')
            if os.name=='nt':
                import msvcrt
                def acquire():
                    handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                def release():
                    handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                def acquire():fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                def release():fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
            while True:
                self._cancel_check()
                try:
                    acquire()
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES,errno.EAGAIN,errno.EDEADLK) and getattr(exc,'winerror',None) not in (33,36):raise
                    # Waiting for another worker is not this controller's GPU
                    # throttle and must not count toward its idle credit.
                    self._sleep_fn(.01)
            try:
                yield
            finally:
                release()
                # A busy inference loop can otherwise reopen/reacquire the lock
                # before the other worker's 10 ms contention poll wakes. Yield
                # outside the OS lock so CPU-heavy line fitting gets GPU turns.
                self._sleep_fn(.012)

    def _cancel_check(self):
        if self._critical_reason is not None:raise ThermalLimitError(self._critical_reason)
        if self.stop_event is not None and self.stop_event.is_set():raise RuntimeCancelledError('GPU work cancelled; save the current checkpoint')

    def _sleep(self,seconds):
        remaining=max(0.,float(seconds))
        while remaining>1e-9:
            self._cancel_check();step=min(.25,remaining);self._sleep_fn(step);remaining-=step
        self.total_throttle_seconds+=max(0.,float(seconds))

    def _sync(self):
        if self._sync_fn is not None:self._sync_fn();return
        import torch
        if torch.cuda.is_available():torch.cuda.synchronize()

    def _sample(self,force=False):
        now=self._clock()
        if not force and self._sampled_monotonic is not None and now-self._sampled_monotonic<self.telemetry_interval_s:return
        self._sampled_monotonic=now
        try:
            sample=self.reader.read()
            if isinstance(sample,dict):sample=GpuTelemetry(**sample)
            self.last_sample=sample;self.last_error=None;self.telemetry_failures=0
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            self.telemetry_failures+=1;self.last_error=str(exc);self.state='telemetry_degraded';self.effective_duty=min(self.duty,.8)
            if self.telemetry_failures>=self.max_telemetry_failures:raise TelemetryUnavailableError(f'GPU telemetry failed {self.telemetry_failures} times: {exc}') from exc
            return
        limits=getattr(self.reader,'hardware_limits_c',{})
        critical=min(self.critical_temperature_c,limits.get('shutdown',float('inf'))-5)
        if sample.temperature_c>=critical:
            self.state='critical_stopped';self._critical_reason=f'GPU temperature {sample.temperature_c:.1f}C reached critical limit {critical:.1f}C';raise ThermalLimitError(self._critical_reason)
        utilization=sample.utilization_percent;self._utilization_samples.append(float(utilization));self._utilization_samples=self._utilization_samples[-12:]
        if self.work_calls and utilization>self.target_high:
            self.duty=max(.25,self.duty-.015-(utilization-self.target_high)*.002)
        elif self.work_calls and utilization<self.target_low:
            self.duty=min(.995,self.duty+.015+(self.target_low-utilization)*.001)
        if self.process_lock_file:self.duty=min(self.duty,self.target_high/100.)
        self.effective_duty=self.duty
        if sample.temperature_c>=self.soft_temperature_c:
            fraction=(sample.temperature_c-self.soft_temperature_c)/(critical-self.soft_temperature_c)
            self.effective_duty=min(self.effective_duty,max(.35,.80-.35*fraction))
        self.state='running'

    def _cool_if_needed(self):
        if self.last_sample is None or self.last_sample.temperature_c<self.pause_temperature_c:return
        started=self._clock();self.state='cooling'
        while self.last_sample.temperature_c>self.soft_temperature_c:
            if self._clock()-started>=self.max_cooling_seconds:
                self.state='cooling_timeout';self._critical_reason='GPU did not cool within the configured pause; preserve checkpoint before retry';raise ThermalLimitError(self._critical_reason)
            self._sleep(2.);self._sample(force=True);self.state='cooling'
        self.state='running'

    def after_gpu_work(self,work_seconds):
        with self._lock:return self._after_gpu_work(work_seconds)

    def _after_gpu_work(self,work_seconds):
        self._cancel_check();seconds=float(work_seconds)
        if not math.isfinite(seconds) or seconds<0:raise ValueError('work_seconds must be finite and nonnegative')
        now=self._clock()
        # In shared mode another process can use the GPU during our CPU work.
        external_gap=0. if self.process_lock_file or self._last_return is None else max(0.,now-self._last_return-seconds)
        self.total_gpu_work_seconds+=seconds;self.work_calls+=1
        self._sample();self._cool_if_needed()
        idle=max(0.,seconds*(1/self.effective_duty-1)-external_gap)
        self._sleep(idle);self._last_return=self._clock()
        return idle

    def throttle(self,work_seconds):
        return self.after_gpu_work(work_seconds)

    @contextmanager
    def gpu_work(self):
        with self._lock:
            if self._inside:
                # Acquiring a second file handle's lock can deadlock this worker.
                yield
            else:
                with self._process_gate():
                    with self._gpu_work_locked():yield

    @contextmanager
    def _gpu_work_locked(self):
        if self._inside:
            # No duplicate synchronization/throttle for a nested helper.
            yield
            return
        self._cancel_check();self._sample();self._cool_if_needed();self._sync()
        started=self._clock();self._inside=True
        try:
            yield
            self._sync()
        except BaseException:
            raise  # Never turn an OOM/model exception into a CPU fallback.
        else:
            self.after_gpu_work(self._clock()-started)
        finally:
            self._inside=False

    def wrap_forward(self,model):
        if getattr(model,'_runtime_control_wrapped',False):return model
        original=model.forward;controller=self
        def controlled_forward(*args,**kwargs):
            with controller.gpu_work():return original(*args,**kwargs)
        model.forward=controlled_forward;model._runtime_control_wrapped=True
        return model

    def before_gpu_frame(self):
        """Fallback prehook for loops that cannot expose their model forward."""
        now=self._clock()
        if self._frame_start is not None:self.after_gpu_work(max(0.,now-self._frame_start))
        else:self._sample();self._cool_if_needed()
        self._frame_start=self._clock()

    def snapshot(self,refresh=False):
        with self._lock:return self._snapshot_locked(refresh)

    def _snapshot_locked(self,refresh=False):
        if refresh:self._sample(force=True)
        return dict(state=self.state,target_utilization_percent=[self.target_low,self.target_high],instantaneous_cap_guaranteed=False,
                    process_gate_active=self.process_lock_file is not None,process_lock_file=self.process_lock_file,
                    duty_target=self.duty,effective_duty=self.effective_duty,telemetry=None if self.last_sample is None else asdict(self.last_sample),
                    hardware_temperature_limits_c=dict(getattr(self.reader,'hardware_limits_c',{})),critical_temperature_c=self.critical_temperature_c,
                    telemetry_failures=self.telemetry_failures,telemetry_error=self.last_error,work_calls=self.work_calls,
                    gpu_work_seconds=self.total_gpu_work_seconds,throttle_seconds=self.total_throttle_seconds,
                    sampled_average_utilization_percent=(sum(self._utilization_samples)/len(self._utilization_samples)) if self._utilization_samples else None,
                    critical_stop_latched=self._critical_reason is not None,
                    model_quality_settings_changed=False)
