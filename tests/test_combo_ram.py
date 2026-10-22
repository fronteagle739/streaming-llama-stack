import ctypes
import hashlib
import subprocess
import sys
import unittest
from slstack.combo_ram import LockedRam, kernel_api


@unittest.skipUnless(sys.platform=='win32','Windows shared RAM contract')
class LockedRamTest(unittest.TestCase):
    def test_locked_shared_pages_and_bounded_transfer(self):
        with LockedRam(8<<20,session='1'*32,host_budget_gib=.5,min_available_gib=4) as ram:
            ram.write(0,b'checkpoint-owned-RAM')
            k=kernel_api();handle=k.OpenFileMappingW(6,False,ram.name)
            self.assertTrue(handle);address=k.MapViewOfFile(handle,6,0,0,ram.capacity)
            self.assertTrue(address)
            try:
                self.assertEqual(ctypes.string_at(address,20),b'checkpoint-owned-RAM')
                ctypes.memmove(address+20,b'!',1)
                self.assertEqual(ram.read(0,21),b'checkpoint-owned-RAM!')
            finally:k.UnmapViewOfFile(address);k.CloseHandle(handle)
            self.assertTrue(ram.locked)
            self.assertEqual(ram.sha256(0,21),hashlib.sha256(b'checkpoint-owned-RAM!').hexdigest())
            for offset,count in ((0,65537),(ram.capacity,1),(-1,4)):
                with self.assertRaises(ValueError):ram.read(offset,count)
            with ram.lease(),self.assertRaises(RuntimeError):ram.close()
        self.assertIsNone(ram.address)
        self.assertIsNone(ram.handle)
        with self.assertRaises(RuntimeError):ram.read(0,1)

    def test_budget_and_floor_refuse_before_allocation(self):
        with self.assertRaises(MemoryError):LockedRam(1<<20,session='1'*32,host_budget_gib=.0001,min_available_gib=4)
        with self.assertRaises(MemoryError):LockedRam(1<<20,session='1'*32,host_budget_gib=.5,min_available_gib=64)

    def test_cross_process_reads_the_same_section(self):
        with LockedRam(1<<20,session='2'*32,host_budget_gib=.5,min_available_gib=4) as ram:
            ram.write(123,b'native-to-ultra')
            program='from slstack.combo_ram import kernel_api;import ctypes,sys;k=kernel_api();h=k.OpenFileMappingW(4,False,sys.argv[1]);assert h;p=k.MapViewOfFile(h,4,0,0,1048576);assert p;assert ctypes.string_at(p+123,15)==b"native-to-ultra";assert k.UnmapViewOfFile(p);assert k.CloseHandle(h);print("SHARED_RAM_CHILD_PASS")'
            result=subprocess.run([sys.executable,'-B','-c',program,ram.name],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('SHARED_RAM_CHILD_PASS',result.stdout)


if __name__=='__main__':unittest.main()
