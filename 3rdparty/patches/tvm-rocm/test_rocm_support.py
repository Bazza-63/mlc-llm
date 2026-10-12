"""Tests for tvm/support/rocm.py path discovery and bitcode selection, with tvm stubbed out."""
import importlib.util
import os
import stat
import sys
import types

import pytest

ROCM_PY = os.environ["ROCM_PY"]


def load_rocm_module(llvm_major=24, device_arch=None):
    tvm = types.ModuleType("tvm")
    tvm.runtime = types.SimpleNamespace(convert=lambda x: list(x))
    tvm.target = types.SimpleNamespace(
        codegen=types.SimpleNamespace(llvm_version_major=lambda allow_none=True: llvm_major)
    )

    class Dev:
        exist = device_arch is not None
        index = 0

        def dlpack_device_type(self):
            return 10

        def _GetDeviceAttr(self, *_):
            return device_arch

    tvm.rocm = lambda i=0: Dev()
    ffi = types.ModuleType("tvm_ffi")
    ffi.register_global_func = lambda name: (lambda f: f)
    sys.modules["tvm"] = tvm
    sys.modules["tvm.runtime"] = tvm.runtime
    sys.modules["tvm.target"] = tvm.target
    sys.modules["tvm_ffi"] = ffi
    pkg = types.ModuleType("tvm.support")
    pkg.__path__ = []
    sys.modules["tvm.support"] = pkg
    utils = types.ModuleType("tvm.support.utils")
    sys.modules["tvm.support.utils"] = utils
    spec = importlib.util.spec_from_file_location("tvm.support.rocm", ROCM_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def touch(path, executable=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    if executable:
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


BITCODE_101 = [
    "ocml", "ockl", "oclc_finite_only_off", "oclc_finite_only_on", "oclc_unsafe_math_off",
    "oclc_unsafe_math_on", "oclc_wavefrontsize64_on", "oclc_wavefrontsize64_off",
    "oclc_isa_version_1101", "oclc_isa_version_942", "oclc_abi_version_400",
    "oclc_abi_version_500", "oclc_abi_version_600",
]


@pytest.fixture
def therock(tmp_path, monkeypatch):
    """TheRock / ROCm 10.1 layout, configured like the Windows reference machine."""
    root = tmp_path / "TheRock" / "build"
    bc = root / "lib" / "llvm" / "amdgcn" / "bitcode"
    for name in BITCODE_101:
        touch(str(bc / (name + ".bc")))
    touch(str(root / "lib" / "llvm" / "bin" / "ld.lld"), executable=True)
    touch(str(root / "bin" / "hipconfig"), executable=True)
    for var in ("ROCM_PATH", "HIP_PATH", "HIP_DEVICE_LIB_PATH", "LLVM_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    return root


def names(files):
    return [os.path.basename(f) for f in files]


def test_find_rocm_path_order(therock, monkeypatch, tmp_path):
    rocm = load_rocm_module()
    monkeypatch.setenv("HIP_PATH", str(therock))
    monkeypatch.setenv("ROCM_PATH", "")  # cleared with setx VAR "" -> unset
    assert rocm.find_rocm_path() == str(therock)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("ROCM_PATH", str(other))
    assert rocm.find_rocm_path() == str(other)


def test_find_rocm_path_from_path_only(therock, monkeypatch):
    rocm = load_rocm_module()
    monkeypatch.setenv("PATH", str(therock / "bin") + os.pathsep + "/usr/bin")
    assert os.path.realpath(rocm.find_rocm_path()) == os.path.realpath(str(therock))


def test_bitcode_dir_layouts(therock, monkeypatch):
    rocm = load_rocm_module()
    monkeypatch.setenv("HIP_PATH", str(therock))
    assert rocm.find_rocm_bitcode_dir() == str(therock / "lib" / "llvm" / "amdgcn" / "bitcode")
    monkeypatch.setenv("HIP_DEVICE_LIB_PATH", str(therock / "lib" / "llvm" / "amdgcn" / "bitcode"))
    assert rocm.find_rocm_bitcode_dir().endswith("bitcode")


def test_bitcode_dir_legacy_layout(tmp_path, monkeypatch):
    rocm = load_rocm_module()
    root = tmp_path / "opt_rocm_6"
    touch(str(root / "amdgcn" / "bitcode" / "ocml.bc"))
    monkeypatch.delenv("HIP_DEVICE_LIB_PATH", raising=False)
    monkeypatch.setenv("ROCM_PATH", str(root))
    assert rocm.find_rocm_bitcode_dir() == str(root / "amdgcn" / "bitcode")


def test_bitcode_rdna_wave32(therock, monkeypatch):
    rocm = load_rocm_module()
    monkeypatch.setenv("HIP_PATH", str(therock))
    files = names(rocm.callback_rocm_bitcode_path("gfx1101", 32))
    assert files == [
        "ocml.bc", "oclc_finite_only_off.bc", "oclc_unsafe_math_off.bc",
        "oclc_wavefrontsize64_off.bc", "oclc_isa_version_1101.bc", "oclc_abi_version_500.bc",
    ]


def test_bitcode_defaults_from_arch(therock, monkeypatch):
    rocm = load_rocm_module()
    monkeypatch.setenv("HIP_PATH", str(therock))
    assert "oclc_wavefrontsize64_off.bc" in names(rocm.callback_rocm_bitcode_path("gfx1101:xnack-"))
    cdna = names(rocm.callback_rocm_bitcode_path("gfx942"))
    assert "oclc_wavefrontsize64_on.bc" in cdna and "oclc_isa_version_942.bc" in cdna


def test_bitcode_one_variant_per_setting(therock, monkeypatch):
    rocm = load_rocm_module()
    monkeypatch.setenv("HIP_PATH", str(therock))
    files = names(rocm.callback_rocm_bitcode_path("gfx1101", 32))
    for flag in ("finite_only", "unsafe_math", "wavefrontsize64", "abi_version", "isa_version"):
        assert sum(flag in f for f in files) == 1, flag


def test_bitcode_missing_isa_warns(therock, monkeypatch):
    rocm = load_rocm_module()
    monkeypatch.setenv("HIP_PATH", str(therock))
    with pytest.warns(UserWarning, match="gfx1030"):
        rocm.callback_rocm_bitcode_path("gfx1030", 32)


def test_bitcode_arch_from_device(therock, monkeypatch):
    rocm = load_rocm_module(device_arch="gfx1101:sramecc-:xnack-")
    monkeypatch.setenv("HIP_PATH", str(therock))
    assert rocm.get_rocm_arch() == "gfx1101"
    assert "oclc_isa_version_1101.bc" in names(rocm.callback_rocm_bitcode_path())


def test_bitcode_legacy_daz_files_used_when_present(tmp_path, monkeypatch):
    rocm = load_rocm_module()
    bc = tmp_path / "r6" / "amdgcn" / "bitcode"
    for name in ["ocml", "oclc_daz_opt_on", "oclc_daz_opt_off", "oclc_correctly_rounded_sqrt_on",
                 "oclc_correctly_rounded_sqrt_off", "oclc_isa_version_1100", "oclc_abi_version_500"]:
        touch(str(bc / (name + ".bc")))
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "r6"))
    monkeypatch.delenv("HIP_DEVICE_LIB_PATH", raising=False)
    files = names(rocm.callback_rocm_bitcode_path("gfx1100", 32))
    assert "oclc_daz_opt_on.bc" in files and "oclc_daz_opt_off.bc" not in files
    assert "oclc_correctly_rounded_sqrt_off.bc" in files
    assert "oclc_correctly_rounded_sqrt_on.bc" not in files


def test_find_lld_prefers_rocm(therock, monkeypatch):
    rocm = load_rocm_module()
    monkeypatch.setenv("HIP_PATH", str(therock))
    assert rocm.find_lld()[0] == str(therock / "lib" / "llvm" / "bin" / "ld.lld")
    monkeypatch.setenv("LLVM_PATH", str(therock / "lib" / "llvm"))
    assert rocm.find_lld()[0] == str(therock / "lib" / "llvm" / "bin" / "ld.lld")


def test_gfx_major():
    rocm = load_rocm_module()
    assert [rocm._gfx_major(x) for x in ["1101", "90a", "942", "900", "1250", "11-generic", "9"]] == [
        11, 9, 9, 9, 12, 11, 9]
