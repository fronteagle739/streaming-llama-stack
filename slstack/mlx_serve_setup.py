"""Build the pinned mlx-serve server without modifying an upstream checkout."""
from pathlib import Path
from .backend_cache import tree_files


def portable_sources(files):
    result=[]
    for name,data in files:
        if name=='build.zig':
            text=data.decode()
            anchor='    const optimize = b.standardOptimizeOption(.{});'
            assert text.count(anchor)==1,'mlx-serve build interface changed'
            text=text.replace(anchor,anchor+'''
    const stack_prefix = b.option([]const u8, "stack-deps-prefix", "Stack-owned dependency stage");
    const stack_include = if (stack_prefix) |p| b.fmt("{s}/include", .{p}) else "/opt/homebrew/include";
    const stack_lib = if (stack_prefix) |p| b.fmt("{s}/lib", .{p}) else "/opt/homebrew/lib";
''')
            text=text.replace('verifyBrewDeps(b);','verifyBrewDeps(b, stack_prefix);',1)
            header='.{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }'
            assert text.count(header)==2
            text=text.replace(header,'.{ .cwd_relative = b.fmt("{s}/webp/decode.h", .{stack_include}) }, .{ .cwd_relative = stack_include }')
            text=text.replace('addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" })','addIncludePath(.{ .cwd_relative = stack_include })')
            text=text.replace('addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" })','addLibraryPath(.{ .cwd_relative = stack_lib })')
            text=text.replace('linkSystemLibrary("webp", .{})','linkSystemLibrary("webp", .{ .use_pkg_config = .no })')
            anchor='fn verifyBrewDeps(b: *std.Build) void {'
            assert text.count(anchor)==1
            text=text.replace(anchor,'''fn verifyBrewDeps(b: *std.Build, stack_prefix: ?[]const u8) void {
    if (stack_prefix) |prefix| {
        buildRootHandle(b).access(b.graph.io, b.fmt("{s}/include/webp/decode.h", .{prefix}), .{}) catch {
            std.debug.print("Stack webp stage is missing; rerun setup-backend.\\n", .{});
            std.process.exit(1);
        };
        return;
    }
''')
            data=text.encode()
        result.append((name,data))
    return result


def install(build):
    base=build.source('mlx-serve')
    libraries={'mlx-serve-ds4':'lib/ds4','mlx-serve-opencode':'lib/opencode2-mlx-serve'}
    combined=tree_files(base,exclude={'stack-source.json'})
    for name,destination in libraries.items():
        combined.extend((destination+'/'+n,d) for n,d in tree_files(build.source(name),exclude={'stack-source.json'}))
    source,revision=build.stage(base,'mlx-serve',transform=lambda _:portable_sources(combined))
    output=build.runtime/'build'/('mlx-serve-'+revision[:16]);output.mkdir(parents=True,exist_ok=True)
    prefix=output/'deps'
    webp,_=build.stage(build.source('webp'),'webp')
    build.cmake(webp,output/'webp',['-DCMAKE_INSTALL_PREFIX='+str(prefix),'-DBUILD_SHARED_LIBS=ON',
                '-DCMAKE_INSTALL_NAME_DIR='+str(prefix/'lib'),'-DWEBP_BUILD_ANIM_UTILS=OFF','-DWEBP_BUILD_CWEBP=OFF',
                '-DWEBP_BUILD_DWEBP=OFF','-DWEBP_BUILD_GIF2WEBP=OFF','-DWEBP_BUILD_IMG2WEBP=OFF',
                '-DWEBP_BUILD_VWEBP=OFF','-DWEBP_BUILD_WEBPINFO=OFF','-DWEBP_BUILD_EXTRAS=OFF'],install=True)
    mlx,_=build.stage(build.source('mlx-serve-mlx'),'mlx-serve-mlx')
    stage=source/'lib/mlx'
    metal_cpp=build.source('metal-cpp');json_source=build.source('mlx-json');fmt=build.source('mlx-fmt')
    build.cmake(mlx,output/'mlx',['-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2','-DBUILD_SHARED_LIBS=ON',
                '-DMLX_BUILD_TESTS=OFF','-DMLX_BUILD_EXAMPLES=OFF','-DCMAKE_INSTALL_PREFIX='+str(stage),
                '-DFETCHCONTENT_SOURCE_DIR_METAL_CPP='+str(metal_cpp),'-DFETCHCONTENT_SOURCE_DIR_JSON='+str(json_source),
                '-DFETCHCONTENT_SOURCE_DIR_FMT='+str(fmt),'-DFETCHCONTENT_FULLY_DISCONNECTED=ON'],install=True)
    mlxc,_=build.stage(build.source('mlx-serve-mlxc'),'mlx-serve-mlxc')
    build.cmake(mlxc,output/'mlxc',['-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2','-DBUILD_SHARED_LIBS=ON',
                '-DMLX_C_USE_SYSTEM_MLX=ON','-DMLX_C_BUILD_EXAMPLES=OFF','-DCMAKE_PREFIX_PATH='+str(stage),
                '-DCMAKE_INSTALL_PREFIX='+str(stage)],install=True)
    pins=build.cache.items
    (stage/'.version').write_text('mlx='+pins['mlx-serve-mlx']['revision'][:12]+' mlxc='+pins['mlx-serve-mlxc']['revision'][:12]+' target=26.2\n')
    if b'_nax' not in (stage/'lib/mlx.metallib').read_bytes():raise RuntimeError('MLX metallib omitted required NAX build inputs')
    llama,_=build.stage(build.source('mlx-serve-llama'),'mlx-serve-llama')
    llama_stage=source/'lib/llama'
    build.cmake(llama,output/'llama',['-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF',
                '-DLLAMA_BUILD_TOOLS=OFF','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_BUILD_COMMON=OFF',
                '-DLLAMA_CURL=OFF','-DLLAMA_OPENSSL=OFF','-DGGML_METAL=ON','-DGGML_METAL_EMBED_LIBRARY=ON',
                '-DBUILD_SHARED_LIBS=ON','-DCMAKE_INSTALL_PREFIX='+str(llama_stage)],install=True)
    (llama_stage/'.version').write_text('b10809\n')
    jinja=source/'lib/jinja_cpp';objects=output/'jinja';objects.mkdir(exist_ok=True)
    for name in ('jinja_wrapper','caps','lexer','parser','runtime','jinja_string','value'):
        build.run(['clang++','-std=c++17','-O2','-DNDEBUG','-I',jinja,'-c',jinja/(name+'.cpp'),'-o',objects/(name+'.o')],'jinja-'+name)
    build.run(['ar','rcs',jinja/'libjinja.a',*[objects/(name+'.o') for name in ('jinja_wrapper','caps','lexer','parser','runtime','jinja_string','value')]],'jinja-archive')
    build.run([build.python,'-B','-m','pip','install',*build.pip_options,'ziglang==0.17.0'],'zig-toolchain')
    build.run([build.python,'-B','-m','ziglang','build','-Doptimize=ReleaseFast','-j'+str(build.jobs),
               '-Dstack-deps-prefix='+str(prefix),'-Dds4-commit='+pins['mlx-serve-ds4']['revision'][:12]],'mlx-serve-compile',cwd=source)
    binary=source/'zig-out/bin/mlx-serve'
    artifacts=[binary,*[p for p in (source/'lib').rglob('*') if p.suffix in {'.dylib','.metallib'} and p.is_file()]]
    return build.report('mlx-serve',revision,artifacts,binary=str(binary),runtime_libraries=str(source/'lib'),
                        compiler='ziglang==0.17.0',deployment_target='26.2')
