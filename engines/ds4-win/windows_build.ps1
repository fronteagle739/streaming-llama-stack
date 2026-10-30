param(
    [ValidateSet("cuda", "cpu")]
    [string]$Target = "cuda",
    [string]$BuildTemp = $PSScriptRoot,
    [ValidatePattern('^[0-9]+[a-z]?$')]
    [string]$CudaArchitecture = $(if ($env:SLSTACK_CUDA_ARCHITECTURES) { $env:SLSTACK_CUDA_ARCHITECTURES } else { '120a' }),
    [switch]$Draft3070,
    [switch]$TestsOnly,
    # ds4-decode.dll: the core + CUDA objects with DS4_EXTERNAL_DECODE=1, loaded by the
    # single-engine ULTRA producer for its decode steps. Objects are rebuilt in place.
    [switch]$ExternalDecode
)

$ErrorActionPreference = "Stop"

# Keep compiler intermediates and the reviewable command file with this checkout.
if (!(Test-Path -LiteralPath $BuildTemp -PathType Container)) {
    throw "BuildTemp must name an existing build directory: $BuildTemp"
}
$env:TEMP = (Resolve-Path -LiteralPath $BuildTemp).Path
$env:TMP = $env:TEMP

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (!(Test-Path $vswhere)) {
    throw "vswhere.exe was not found. Install Visual Studio Build Tools with the C++ workload."
}

$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vsPath) {
    throw "Visual Studio C++ Build Tools were not found."
}

# [2026-08-20 FIX] Prefer vcvars64.bat over VsDevCmd.bat.
#
# On this box (VS 18 BuildTools) `call VsDevCmd.bat -arch=x64` dies with
#     'vswhere.exe' is not recognized as an internal or external command
# from INSIDE VsDevCmd, even though vswhere is present and this script's own top-level lookup
# resolved it by full path a few lines above. VsDevCmd re-resolves vswhere through its own PATH
# assumptions, and those do not hold in the `cmd /d /s /c` context this script builds.
#
# vcvars64.bat is self-contained and works here (verified: `where cl.exe` resolves to
# VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe). It is x64 by definition, so it takes NO -arch
# argument - passing one is what would silently give a host-x86 toolchain.
# BOTH vcvars64.bat and VsDevCmd.bat invoke `vswhere.exe` BARE, relying on it being on PATH. It is
# not, in the `cmd /d /s /c` context this script creates, so the call fails with
#     'vswhere.exe' is not recognized as an internal or external command
# and the `if errorlevel 1` immediately after aborts the build before a single file is compiled.
# The message says precisely what is wrong, so the fix is to put the Installer directory on PATH for
# the child cmd rather than to reverse-engineer MSVC's batch internals.
$vswhereDir = Split-Path -Parent $vswhere
$vsEnvCmd = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
$vsEnvArgs = ""
if (!(Test-Path $vsEnvCmd)) {
    $vsEnvCmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
    $vsEnvArgs = " -arch=x64"
    if (!(Test-Path $vsEnvCmd)) {
        throw "Neither vcvars64.bat nor VsDevCmd.bat was found under $vsPath"
    }
    Write-Host "vcvars64.bat absent; falling back to VsDevCmd.bat" -ForegroundColor Yellow
}
Write-Host "VS environment: $vsEnvCmd$vsEnvArgs"

$defines = @(
    "/D_CRT_SECURE_NO_WARNINGS",
    "/D_CRT_NONSTDC_NO_DEPRECATE",
    "/DWIN32_LEAN_AND_MEAN",
    "/DNOMINMAX",
    "/utf-8"
)
if ($Target -eq "cpu") {
    $defines += "/DDS4_NO_GPU"
}
if ($ExternalDecode) {
    if ($Target -ne "cuda" -or $TestsOnly) { throw "-ExternalDecode builds the CUDA DLL only" }
    $defines += "/DDS4_EXTERNAL_DECODE=1"
}

# [merge 2026-08-15] ds4_gpu_args.c, ds4_tp.c and ds4_layer_pack.c are NEW upstream sources this
# Windows build script had never heard of, so the link failed with 26 LNK2019 unresolved externals
# (parse_gpu_vram_arg, format_gpu_layout_line, ds4_tp_* ...). Note every one of these files compiled
# CLEANLY on its own -- a per-file syntax check cannot catch a source that is simply never built.
# Deliberately NOT added: ds4_distributed.c and linenoise.c are the POSIX variants, superseded here
# by ds4_distributed_win32.c and linenoise_win32.c; ds4_eval.c defines none of the missing symbols.
#
# [2026-08-16] ds4_tp.c joins that POSIX-only list: it needs netdb.h, sys/uio.h, netinet/in.h,
# netinet/tcp.h, pthread.h and stdatomic.h, so MSVC stops at C1083 on the first include. It is
# replaced by ds4_tp_win32.c, which defines the same 42 externals as stubs that FAIL LOUDLY --
# tensor parallelism is a two-machine feature and this is a single-GPU box, so the honest answer
# is a clear error, never a silent no-op that would look like TP was running.
$coreSources = @(
    "ds4.c",
    "ds4_help.c",
    "ds4_ssd.c",
    "ds4_win32.c",
    "ds4_distributed_win32.c",
    "ds4_gpu_args.c",
    "ds4_tp_win32.c",
    "ds4_layer_pack.c",
    "ds4_image.c",
    "ds4_prompt_prefix.c",
    "ds4_engram.c"
)

$cliSources = @(
    "ds4_cli.c",
    "linenoise_win32.c"
)

$benchSources = @(
    "ds4_bench.c"
)

$serverSources = @(
    "ds4_server.c",
    "ds4_peer.c",
    "ds4_kvstore.c",
    "rax.c"
)

$agentSources = @(
    "ds4_agent.c",
    "ds4_web.c",
    "ds4_kvstore.c",
    "linenoise_win32.c"
)

$sources = if ($ExternalDecode) {
    $coreSources
} elseif ($TestsOnly) {
    # Compile the data-plane owner too, but never link ds4-server.exe: a live
    # Studio server keeps that binary locked while Phase 3's standalone gates
    # need only core objects plus ds4_peer.c.
    $coreSources + $serverSources
} else {
    $coreSources + $cliSources + $benchSources + $serverSources + $agentSources
}
$sources = $sources | Select-Object -Unique
$common = @("/nologo", "/std:c11", "/O2", "/Zi", "/W3", "/wd4244", "/wd4267", "/wd4996") + $defines
$compile = "cl " + (($common + "/c" + $sources) -join " ")
$payloadTestSource = "tests\test_kv_payload_roundtrip.c"
$payloadTestObject = "test_kv_payload_roundtrip.obj"
$payloadTestCompile = "cl " + (($common + "/I." + "/c" + "/Fo$payloadTestObject" + $payloadTestSource) -join " ")
$g3TransportTestSource = "tests\test_ds4_peer_transport_g3.c"
$g3TransportTestObject = "test_ds4_peer_transport_g3.obj"
$g3TransportTestCompile = "cl " + (($common + "/I." + "/c" + "/Fo$g3TransportTestObject" + $g3TransportTestSource) -join " ")
$g56SessionTestSource = "tests\test_ds4_peer_sessions_g56.c"
$g56SessionTestObject = "test_ds4_peer_sessions_g56.obj"
$g56SessionTestCompile = "cl " + (($common + "/I." + "/c" + "/Fo$g56SessionTestObject" + $g56SessionTestSource) -join " ")
$g3TopologyTestSource = "tests\test_ds4_peer_topology_a.c"
$g3TopologyTestObject = "test_ds4_peer_topology_a.obj"
$g3TopologyTestCompile = "cl " + (($common + "/I." + "/c" + "/Fo$g3TopologyTestObject" + $g3TopologyTestSource) -join " ")
$coreObjects = $coreSources | ForEach-Object { [IO.Path]::ChangeExtension($_, ".obj") }
$cliObjects = $cliSources | ForEach-Object { [IO.Path]::ChangeExtension($_, ".obj") }
$benchObjects = $benchSources | ForEach-Object { [IO.Path]::ChangeExtension($_, ".obj") }
$serverObjects = $serverSources | ForEach-Object { [IO.Path]::ChangeExtension($_, ".obj") }
$agentObjects = $agentSources | ForEach-Object { [IO.Path]::ChangeExtension($_, ".obj") }
$backendObjects = @("ds4_lora.obj", "ds4_engram_ioring.obj", "ds4_cpu_fp4.obj")
$win32Link = " ws2_32.lib bcrypt.lib"

$loraCommand = "cl " + ((@("/nologo", "/std:c++17", "/EHsc", "/O2", "/W3") + $defines + "/c" + "ds4_lora.cpp") -join " ")
# Engram raw-row reads through IoRing (ds4_engram.c calls it on Windows); APIs resolved at runtime.
$engramIoringCommand = "cl " + ((@("/nologo", "/std:c++17", "/EHsc", "/O2", "/W3") + $defines + "/c" + "ds4_engram_ioring.cpp") -join " ")
# RAM-tier packed FP4 experts on the CPU (AVX-512 VNNI); its own unit so only it gets /arch:AVX512.
$engramIoringCommand += "`nif errorlevel 1 exit /b %errorlevel%`ncl " + ((@("/nologo", "/std:c++17", "/EHsc", "/O2", "/W3", "/arch:AVX512") + $defines + "/c" + "ds4_cpu_fp4.cpp") -join " ")
$cudaCommand = ""
$dstorageCommand = "rem no DirectStorage transport in this target"
$cudaLink = ""
if ($Target -eq "cuda") {
    $cudaPath = $env:CUDA_PATH
    if (!$cudaPath) {
        $nvccCmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
        if ($nvccCmd) {
            $cudaPath = Split-Path -Parent (Split-Path -Parent $nvccCmd.Source)
        }
    }
    if (!$cudaPath -or !(Test-Path (Join-Path $cudaPath "bin\nvcc.exe"))) {
        throw "CUDA Toolkit was not found. Set CUDA_PATH or build with -Target cpu."
    }
    $nvcc = Join-Path $cudaPath "bin\nvcc.exe"
    $cudaLib = Join-Path $cudaPath "lib\x64"
    $cudaInclude = Join-Path $cudaPath "include"
    if (!(Test-Path $cudaLib)) {
        throw "CUDA library directory was not found: $cudaLib"
    }
    $nvccDefines = @(
        "-D_CRT_SECURE_NO_WARNINGS",
        "-D_CRT_NONSTDC_NO_DEPRECATE",
        "-DWIN32_LEAN_AND_MEAN",
        "-DNOMINMAX",
        "-DDS4_CUDA_HAVE_MXF4=1"
    )
    if ($ExternalDecode) { $nvccDefines += "-DDS4_EXTERNAL_DECODE=1" }
    # Target the GPU this tree is built for. WITHOUT these, nvcc uses its DEFAULT architecture, which on
    # CUDA 13.2 is the lowest it still supports: sm_75, Turing, 2018. That was not a theory - cuobjdump
    # on the shipped binary reported 8 cubins, ALL sm_75, plus sm_75 PTX, on an sm_120 card:
    #     cuobjdump --list-elf ds4.exe   ->   ds4.1.sm_75.cubin ... ds4.8.sm_75.cubin
    # So every ds4 kernel ran as PTX JIT-compiled from a Turing target: register allocation, shared
    # memory budgeting and instruction selection all decided for the wrong architecture, and every
    # sm_90+ feature unreachable at compile time (programmatic dependent launch, INT8 and NVFP4
    # tensor-core MMA, cp.async.bulk.tensor TMA, cluster launch). The sibling llama.cpp tree in this
    # same workspace ships 133 sm_120a cubins, so the toolchain was never the limit.
    #
    # compute_120a, NOT plain compute_120. Tried plain 120 first and cuda/mmq/ds4_mmq.cu failed:
    #     ptxas error : Instruction 'mma with block scale' not supported on .target 'sm_120'
    #     ptxas error : Feature '.kind::mxf4' not supported on .target 'sm_120'
    #     ptxas error : Feature '.scale_vec::2X' not supported on .target 'sm_120'
    #     ptxas fatal : Ptx assembly aborted due to errors
    # That file already contains NVFP4/MXFP4 block-scaled MMA, and those instructions exist only on the
    # arch-SPECIFIC target. The 'a' suffix is required, not optional. The sibling llama.cpp tree in this
    # workspace ships 133 sm_120a cubins, so this is the known-good target for this box.
    #
    # Consequence accepted deliberately: sm_120a is NOT forward compatible, so this tree targets Blackwell
    # only. That is correct for ds4-win, which exists for this specific machine. The compute_120a PTX
    # image is kept so a future driver can still JIT.
    #
    # ⚠️ VERIFY AFTER EVERY BUILD - exit 0 proves nothing (feedback_build_error_grep_must_match_nvcc):
    #     cuobjdump --list-elf ds4.exe | grep -oE "sm_[0-9]+a?" | sort | uniq -c
    # must report EIGHT sm_120a and ZERO sm_75. Seven-of-eight is the signature of a masked compile
    # failure leaving a stale object behind - which is exactly what happened on the first attempt.
    $nvccArch = @(
        "-gencode", "arch=compute_${CudaArchitecture},code=sm_${CudaArchitecture}",
        "-gencode", "arch=compute_${CudaArchitecture},code=compute_${CudaArchitecture}"
    )
    # ⚠️ EVERY nvcc invocation gets its OWN errorlevel check. $cudaCommand is emitted into the .cmd as a
    # single multi-line block with one "if errorlevel 1" AFTER it, so without these per-line checks only
    # the LAST compile's status was ever tested. Measured consequence on 2026-08-17: ds4_mmq.cu died with
    # "ptxas fatal : Ptx assembly aborted due to errors", the six files after it succeeded, errorlevel at
    # the end was 0, the link happily reused the STALE sm_75 object from the previous build, and the
    # script exited 0. cuobjdump then showed 7 sm_120 + 1 sm_75 - a silently mixed binary.
    $errCheck = "`nif errorlevel 1 exit /b %errorlevel%"
    $draftArch = @()
    if ($Draft3070) { $draftArch = @("-gencode", "arch=compute_86,code=sm_86") }
    $cudaCommand = "`"$nvcc`" -std=c++17 -Xcompiler /Zc:preprocessor " + (($nvccArch + $draftArch + $nvccDefines + "-c" + "-o" + "ds4_cuda.obj" + "ds4_cuda.cu") -join " ") + $errCheck
    $backendObjects += "ds4_cuda.obj"

    # [merge 2026-08-15] The vendored MMQ kernels are NEW from upstream (cuda/mmq/, added wholesale by
    # this merge) and were never compiled here, leaving ~19 unresolved ds4_mmq_* externals at link.
    # Same class as the three new .c files above: each compiles fine in isolation, but a source that
    # is never built cannot be caught by any per-file check -- only by linking.
    $mmqSources = @(
        "cuda/mmq/ds4_mmq.cu",
        "cuda/mmq/ds4_mmq_d2r.cu",
        "cuda/mmq/ds4_repack.cu",
        "cuda/mmq/ds4_ggml_stubs.cu",
        "cuda/mmq/mmid.cu",
        "cuda/mmq/mmvq.cu",
        "cuda/mmq/quantize.cu",
        "cuda/mmq/ds4_q4k_half.cu"
    )
    foreach ($m in $mmqSources) {
        $obj = "mmq_" + [IO.Path]::GetFileNameWithoutExtension($m) + ".obj"
        $cudaCommand += "`n`"$nvcc`" -std=c++17 -Xcompiler /Zc:preprocessor -Icuda/mmq " + (($nvccArch + $nvccDefines + "-c" + "-o" + $obj + $m) -join " ") + $errCheck
        $backendObjects += $obj
    }
    $cudaLink = " /LIBPATH:`"$cudaLib`" cudart.lib cublas.lib cuda.lib ole32.lib d3d12.lib dxgi.lib"

    # DirectStorage transport. Built by cl as C++17 rather than nvcc, because dstorage.h drags in
    # d3d12.h/unknwn.h which nvcc has no reason to parse. The default queue uses a NULL D3D12 device;
    # The optional GPU-destination proof gate creates a shareable D3D12 resource and imports it into
    # CUDA, so this translation unit also needs the CUDA runtime headers plus d3d12/dxgi at link.
    # dstorage.lib remains unnecessary: dstorage.dll is still resolved at runtime.
    $dsInclude = "external\directstorage\native\include"
    if (!(Test-Path (Join-Path $dsInclude "dstorage.h"))) {
        throw "DirectStorage SDK headers were not found at $dsInclude"
    }
    $dsCommon = @("/nologo", "/std:c++17", "/EHsc", "/O2", "/Zi", "/W3", "/wd4244", "/wd4267") + $defines
    $dstorageCommand = "cl " + (($dsCommon + "/I$dsInclude" + "/I`"$cudaInclude`"" + "/c" + "ds4_dstorage.cpp") -join " ")
    $backendObjects += "ds4_dstorage.obj"
}

$cliLinkObjects = @($cliObjects) + @($coreObjects) + @($backendObjects)
$benchLinkObjects = @($benchObjects) + @($coreObjects) + @($backendObjects)
$serverLinkObjects = @($serverObjects) + @($coreObjects) + @($backendObjects)
$agentLinkObjects = @($agentObjects) + @($coreObjects) + @($backendObjects)
$payloadTestLinkObjects = @($payloadTestObject) + @($coreObjects) + @($backendObjects)
$g3TransportTestLinkObjects = @($g3TransportTestObject, "ds4_peer.obj", "ds4_win32.obj")
$g56SessionTestLinkObjects = @($g56SessionTestObject, "ds4_peer.obj", "ds4_win32.obj")
$g3TopologyTestLinkObjects = @($g3TopologyTestObject, "ds4_peer.obj") + @($coreObjects) + @($backendObjects)
$linkCli = "link /nologo /out:ds4.exe " + ($cliLinkObjects -join " ") + $cudaLink + $win32Link
$linkBench = "link /nologo /out:ds4-bench.exe " + ($benchLinkObjects -join " ") + $cudaLink + $win32Link
$linkServer = "link /nologo /out:ds4-server.exe " + ($serverLinkObjects -join " ") + $cudaLink + $win32Link
$linkAgent = "link /nologo /out:ds4-agent.exe " + ($agentLinkObjects -join " ") + $cudaLink + $win32Link
$linkPayloadTest = "link /nologo /out:test-kv-payload-roundtrip.exe " + ($payloadTestLinkObjects -join " ") + $cudaLink + $win32Link
$linkG3TransportTest = "link /nologo /out:ds4-peer-transport-g3.exe " + ($g3TransportTestLinkObjects -join " ") + $win32Link
$linkG56SessionTest = "link /nologo /out:ds4-peer-sessions-g56.exe " + ($g56SessionTestLinkObjects -join " ") + $win32Link
$linkG3TopologyTest = "link /nologo /out:ds4-peer-topology-a-g3.exe " + ($g3TopologyTestLinkObjects -join " ") + $cudaLink + $win32Link

if ($ExternalDecode) {
    $testCompileSteps = ""
    $linkSteps = "link /nologo /DLL /out:ds4-decode.dll " + ((@($coreObjects) + @($backendObjects)) -join " ") + $cudaLink + $win32Link + "`nif errorlevel 1 exit /b %errorlevel%"
} elseif ($TestsOnly) {
    $testCompileSteps = @"
$g3TransportTestCompile
if errorlevel 1 exit /b %errorlevel%
$g56SessionTestCompile
if errorlevel 1 exit /b %errorlevel%
$g3TopologyTestCompile
if errorlevel 1 exit /b %errorlevel%
"@
    $linkSteps = @"
$linkG3TransportTest
if errorlevel 1 exit /b %errorlevel%
$linkG56SessionTest
if errorlevel 1 exit /b %errorlevel%
$linkG3TopologyTest
if errorlevel 1 exit /b %errorlevel%
"@
} else {
    $testCompileSteps = @"
$payloadTestCompile
if errorlevel 1 exit /b %errorlevel%
"@
    $linkSteps = @"
$linkCli
if errorlevel 1 exit /b %errorlevel%
$linkBench
if errorlevel 1 exit /b %errorlevel%
$linkServer
if errorlevel 1 exit /b %errorlevel%
$linkAgent
if errorlevel 1 exit /b %errorlevel%
$linkPayloadTest
if errorlevel 1 exit /b %errorlevel%
"@
}

$cmd = @"
@echo on
set "PATH=$vswhereDir;%PATH%"
call "$vsEnvCmd"$vsEnvArgs
if errorlevel 1 exit /b %errorlevel%
$compile
if errorlevel 1 exit /b %errorlevel%
$testCompileSteps
$loraCommand
if errorlevel 1 exit /b %errorlevel%
$engramIoringCommand
if errorlevel 1 exit /b %errorlevel%
$cudaCommand
if errorlevel 1 exit /b %errorlevel%
$dstorageCommand
if errorlevel 1 exit /b %errorlevel%
$linkSteps
"@

$cmdPath = Join-Path $env:TEMP "ds4-windows-build-$PID.cmd"
Set-Content -LiteralPath $cmdPath -Encoding ASCII -Value $cmd
try {
    & cmd /d /s /c "`"$cmdPath`""
    $buildExit = $LASTEXITCODE
    if ($buildExit -eq 0 -and $Target -eq "cuda" -and -not $TestsOnly) {
        # App-local DirectStorage redistributables. dstorage.dll loads dstoragecore.dll from its own
        # directory, so both must sit beside the executables; ds4 resolves dstorage.dll by explicit
        # path from the exe directory rather than the default search order.
        $dsBin = "external\directstorage\native\bin\x64"
        Copy-Item (Join-Path $dsBin "dstorage.dll"), (Join-Path $dsBin "dstoragecore.dll") `
            -Destination . -Force
    }
    exit $buildExit
} finally {
    # Keep the exact generated command file with the build evidence.
    Write-Host "Build command retained: $cmdPath"
}
