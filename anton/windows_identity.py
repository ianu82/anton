from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes
from dataclasses import dataclass

from anton.execution_policy import ExecutionMode

ERROR_ALREADY_EXISTS_HRESULT = 0x800700B7
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400
PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020005
STARTF_USESTDHANDLES = 0x00000100
STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12


class STARTUPINFO(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class STARTUPINFOEX(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", STARTUPINFO),
        ("lpAttributeList", ctypes.c_void_p),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("Sid", ctypes.c_void_p),
        ("Attributes", wintypes.DWORD),
    ]


class SECURITY_CAPABILITIES(ctypes.Structure):
    _fields_ = [
        ("AppContainerSid", ctypes.c_void_p),
        ("Capabilities", ctypes.POINTER(SID_AND_ATTRIBUTES)),
        ("CapabilityCount", wintypes.DWORD),
        ("Reserved", wintypes.DWORD),
    ]


@dataclass(frozen=True)
class AppContainerProfile:
    name: str
    sid_ptr: int
    sid_string: str


@dataclass(frozen=True)
class WindowsProcessHandles:
    process_handle: int
    thread_handle: int
    pid: int


def _kernel32():
    if sys.platform != "win32":
        raise RuntimeError("Windows identity helpers are only available on Windows hosts.")
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _advapi32():
    if sys.platform != "win32":
        raise RuntimeError("Windows identity helpers are only available on Windows hosts.")
    return ctypes.WinDLL("advapi32", use_last_error=True)


def _userenv():
    if sys.platform != "win32":
        raise RuntimeError("Windows identity helpers are only available on Windows hosts.")
    return ctypes.WinDLL("userenv", use_last_error=True)


def appcontainer_name_for_mode(mode: ExecutionMode | str) -> str:
    resolved = ExecutionMode.coerce(mode)
    suffix = "WorkspaceWrite" if resolved is ExecutionMode.WORKSPACE_WRITE else "ReadOnly"
    return f"AntonScratchpad{suffix}"


def _check_hresult(hr: int, *, allow_already_exists: bool = False) -> None:
    if hr == 0:
        return
    if allow_already_exists and hr == ERROR_ALREADY_EXISTS_HRESULT:
        return
    raise RuntimeError(f"Windows AppContainer call failed with HRESULT 0x{hr:08x}.")


def _sid_to_string(sid_ptr: int) -> str:
    advapi32 = _advapi32()
    convert = advapi32.ConvertSidToStringSidW
    convert.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    convert.restype = wintypes.BOOL

    kernel32 = _kernel32()
    local_free = kernel32.LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p

    sid_string = wintypes.LPWSTR()
    if not convert(ctypes.c_void_p(sid_ptr), ctypes.byref(sid_string)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return str(sid_string.value)
    finally:
        local_free(sid_string)


def free_sid(sid_ptr: int) -> None:
    advapi32 = _advapi32()
    free = advapi32.FreeSid
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_void_p
    free(ctypes.c_void_p(sid_ptr))


def ensure_appcontainer_profile(mode: ExecutionMode | str) -> AppContainerProfile:
    resolved = ExecutionMode.coerce(mode)
    name = appcontainer_name_for_mode(resolved)
    userenv = _userenv()

    create = userenv.CreateAppContainerProfile
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    create.restype = ctypes.c_long

    derive = userenv.DeriveAppContainerSidFromAppContainerName
    derive.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    derive.restype = ctypes.c_long

    created_sid = ctypes.c_void_p()
    _check_hresult(
        int(create(name, name, name, None, 0, ctypes.byref(created_sid))),
        allow_already_exists=True,
    )
    if created_sid.value:
        free_sid(int(created_sid.value))

    sid_ptr = ctypes.c_void_p()
    _check_hresult(int(derive(name, ctypes.byref(sid_ptr))))
    if not sid_ptr.value:
        raise RuntimeError("Windows AppContainer profile did not return a SID.")
    sid_int = int(sid_ptr.value)
    return AppContainerProfile(name=name, sid_ptr=sid_int, sid_string=_sid_to_string(sid_int))


def _build_environment_block(env: dict[str, str]) -> ctypes.Array[ctypes.c_wchar]:
    items = [f"{key}={value}" for key, value in sorted(env.items())]
    return ctypes.create_unicode_buffer("\0".join(items) + "\0\0")


def _std_handle(kind: int) -> int:
    kernel32 = _kernel32()
    getter = kernel32.GetStdHandle
    getter.argtypes = [ctypes.c_int]
    getter.restype = wintypes.HANDLE
    handle = getter(kind)
    if handle in (0, wintypes.HANDLE(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle)


def launch_appcontainer_process(
    *,
    profile: AppContainerProfile,
    executable: str,
    args: tuple[str, ...],
    cwd: str,
    env: dict[str, str],
) -> WindowsProcessHandles:
    kernel32 = _kernel32()

    init_attr_list = kernel32.InitializeProcThreadAttributeList
    init_attr_list.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.c_size_t)]
    init_attr_list.restype = wintypes.BOOL

    update_attr = kernel32.UpdateProcThreadAttribute
    update_attr.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    update_attr.restype = wintypes.BOOL

    delete_attr_list = kernel32.DeleteProcThreadAttributeList
    delete_attr_list.argtypes = [ctypes.c_void_p]
    delete_attr_list.restype = None

    create_process = kernel32.CreateProcessW
    create_process.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    create_process.restype = wintypes.BOOL

    size = ctypes.c_size_t()
    init_attr_list(None, 1, 0, ctypes.byref(size))
    attr_list = ctypes.create_string_buffer(size.value)
    if not init_attr_list(attr_list, 1, 0, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())

    capabilities = SECURITY_CAPABILITIES(
        AppContainerSid=ctypes.c_void_p(profile.sid_ptr),
        Capabilities=None,
        CapabilityCount=0,
        Reserved=0,
    )
    if not update_attr(
        attr_list,
        0,
        PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
        ctypes.byref(capabilities),
        ctypes.sizeof(capabilities),
        None,
        None,
    ):
        delete_attr_list(attr_list)
        raise ctypes.WinError(ctypes.get_last_error())

    startup = STARTUPINFOEX()
    startup.StartupInfo.cb = ctypes.sizeof(startup)
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES
    startup.StartupInfo.hStdInput = wintypes.HANDLE(_std_handle(STD_INPUT_HANDLE))
    startup.StartupInfo.hStdOutput = wintypes.HANDLE(_std_handle(STD_OUTPUT_HANDLE))
    startup.StartupInfo.hStdError = wintypes.HANDLE(_std_handle(STD_ERROR_HANDLE))
    startup.lpAttributeList = ctypes.cast(attr_list, ctypes.c_void_p)

    process_info = PROCESS_INFORMATION()
    env_block = _build_environment_block(env)
    command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline([executable, *args]))
    flags = EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED

    try:
        ok = create_process(
            executable,
            command_line,
            None,
            None,
            True,
            flags,
            env_block,
            cwd,
            ctypes.byref(startup),
            ctypes.byref(process_info),
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return WindowsProcessHandles(
            process_handle=int(process_info.hProcess),
            thread_handle=int(process_info.hThread),
            pid=int(process_info.dwProcessId),
        )
    finally:
        delete_attr_list(attr_list)
