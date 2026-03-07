from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
WAIT_OBJECT_0 = 0x00000000
INFINITE = 0xFFFFFFFF


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel32():
    if sys.platform != "win32":
        raise RuntimeError("Windows job helpers are only available on Windows.")
    return ctypes.WinDLL("kernel32", use_last_error=True)


def create_kill_on_close_job() -> int:
    kernel32 = _kernel32()
    create_job = kernel32.CreateJobObjectW
    create_job.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    create_job.restype = wintypes.HANDLE

    set_info = kernel32.SetInformationJobObject
    set_info.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    set_info.restype = wintypes.BOOL

    handle = create_job(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())

    limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = set_info(
        handle,
        JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    )
    if not ok:
        err = ctypes.get_last_error()
        close_handle(handle)
        raise ctypes.WinError(err)
    return int(handle)


def assign_process_to_job(job_handle: int, process_handle: int) -> None:
    kernel32 = _kernel32()
    assign = kernel32.AssignProcessToJobObject
    assign.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    assign.restype = wintypes.BOOL
    ok = assign(wintypes.HANDLE(job_handle), wintypes.HANDLE(process_handle))
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())


def close_handle(handle: int) -> None:
    kernel32 = _kernel32()
    closer = kernel32.CloseHandle
    closer.argtypes = [wintypes.HANDLE]
    closer.restype = wintypes.BOOL
    closer(wintypes.HANDLE(handle))


def resume_thread(thread_handle: int) -> None:
    kernel32 = _kernel32()
    resume = kernel32.ResumeThread
    resume.argtypes = [wintypes.HANDLE]
    resume.restype = wintypes.DWORD
    result = resume(wintypes.HANDLE(thread_handle))
    if result == 0xFFFFFFFF:
        raise ctypes.WinError(ctypes.get_last_error())


def wait_for_process_exit(process_handle: int) -> int:
    kernel32 = _kernel32()
    wait = kernel32.WaitForSingleObject
    wait.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    wait.restype = wintypes.DWORD

    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    get_exit_code.restype = wintypes.BOOL

    result = wait(wintypes.HANDLE(process_handle), INFINITE)
    if result != WAIT_OBJECT_0:
        raise ctypes.WinError(ctypes.get_last_error())

    exit_code = wintypes.DWORD()
    if not get_exit_code(wintypes.HANDLE(process_handle), ctypes.byref(exit_code)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(exit_code.value)
