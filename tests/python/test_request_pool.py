import sys,time,unittest
sys.path.insert(0,'python')
from ov203.request_pool import RequestPool
from ov203.device import parse_device_spec

class RequestPoolTests(unittest.TestCase):
    def test_completion_order_keeps_context(self):
        def worker(v):
            time.sleep(.08 if v == 'A' else .005)
            return v
        pool=RequestPool([worker,worker],default_timeout=1)
        try:
            a=object(); b=object(); pool.submit('A',context=a); pool.submit('B',context=b)
            first=pool.get(); second=pool.get()
            self.assertEqual(first.value,'B'); self.assertIs(first.context,b)
            self.assertEqual(second.value,'A'); self.assertIs(second.context,a)
        finally: pool.close()
    def test_worker_failure_is_prompt(self):
        pool=RequestPool([lambda _: (_ for _ in ()).throw(ValueError('boom'))],default_timeout=5)
        try:
            pool.submit(1); t=time.monotonic()
            with self.assertRaisesRegex(RuntimeError,'boom'): pool.get()
            self.assertLess(time.monotonic()-t,1)
            with self.assertRaises(RuntimeError): pool.submit(2)
        finally: pool.close()
    def test_composite_device(self):
        d=parse_device_spec('HETERO:CPU,MYRIAD')
        self.assertTrue(d.uses_cpu and d.uses_myriad); self.assertEqual(d.physical_devices,('CPU','MYRIAD'))

if __name__=='__main__': unittest.main()
