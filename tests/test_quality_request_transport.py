"""Compile the production quality request fields/store/dispatch, without a model."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest
import uuid


def function(source, name):
    found = re.search(r'static (?:int|void) ' + name + r'\(', source)
    if not found:
        raise AssertionError('Missing production function: ' + name)
    start = source.index('{', found.start())
    depth = 0
    for index in range(start, len(source)):
        depth += (source[index] == '{') - (source[index] == '}')
        if depth == 0:
            return source[found.start():index + 1]
    raise AssertionError('Unclosed production function: ' + name)


class QualityRequestTransportTests(unittest.TestCase):
    def test_queued_code_setting_is_independent_and_scalar_compatibility_survives(self):
        root = Path(__file__).resolve().parents[1]
        source_path = os.environ.get('SLSTACK_SERVER_SOURCE')
        if not source_path:
            source_path = next(p for p in (root / 'ds4_server.c', root / 'engines/ds4-win/ds4_server.c') if p.is_file())
        source = Path(source_path).read_text(encoding='utf-8')
        fields = re.search(r'bool quality_speed_set;\s*float quality_speed;\s*bool quality_per_phase;\s*float quality_phases\[\d+\];', source)
        setting = re.search(r'typedef struct \{bool per_phase;float value,phases\[4\];\} quality_setting;', source)
        self.assertIsNotNone(fields)
        self.assertIsNotNone(setting)
        compiler = os.environ.get('SLSTACK_CC') or shutil.which('cl') or shutil.which('cc')
        if not compiler:
            self.skipTest('A C compiler is required for the production request regression')
        temporary = Path(os.environ.get('SLSTACK_TEST_ROOT', root / 'runtime')) / ('quality-request-' + uuid.uuid4().hex)
        temporary.mkdir(parents=True)
        fixture = temporary / 'request.c'
        executable = temporary / ('request.exe' if os.name == 'nt' else 'request')
        program = r'''
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
typedef struct { __FIELDS__ } request;
__SETTING__
static float observed[4];
static int calls;
static int ds4_quality_speed_set(float value) {
    for (int i=0;i<4;i++) observed[i]=value;
    calls++;return 0;
}
static int ds4_quality_speed_set_phase_map(const float values[4]) {
    memcpy(observed,values,sizeof(observed));calls++;return 0;
}
static int ds4_quality_speed_set_phases(const float values[3]) {
    const float map[4]={values[0],values[1],values[2],values[2]};
    return ds4_quality_speed_set_phase_map(map);
}
__STORE__
__APPLY__
int main(void) {
    request queued={0};
    quality_setting chosen={true,.35f,{.65f,0,.35f,1.36125f}};
    request_quality_store(&queued,true,&chosen);
    if(request_quality_apply(&queued))return 2;
    for(int i=0;i<4;i++)if(observed[i]!=chosen.phases[i]) {
        fprintf(stderr,"phase %d lost: received %.9g expected %.9g\n",i,observed[i],chosen.phases[i]);return 1;
    }
    chosen.per_phase=false;chosen.value=.9f;
    request_quality_store(&queued,true,&chosen);
    if(request_quality_apply(&queued))return 3;
    for(int i=0;i<4;i++)if(observed[i]!=.9f)return 4;
    calls=0;request_quality_store(&queued,false,&chosen);
    if(request_quality_apply(&queued) || calls)return 5;
    puts("queued four-phase, scalar and absent-setting request transport: PASS");return 0;
}
'''
        program = program.replace('__FIELDS__', fields[0]).replace('__SETTING__', setting[0])
        program = program.replace('__STORE__', function(source, 'request_quality_store'))
        program = program.replace('__APPLY__', function(source, 'request_quality_apply'))
        fixture.write_text(program, encoding='utf-8')
        try:
            if Path(compiler).stem.lower() == 'cl':
                command = [compiler, '/nologo', '/TC', '/std:c11', '/O2', str(fixture),
                           '/Fe' + str(executable), '/Fo' + str(temporary / 'request.obj'),
                           '/link', '/INCREMENTAL:NO']
            else:
                command = [compiler, '-std=c11', '-O2', str(fixture), '-o', str(executable)]
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        finally:
            for file in temporary.iterdir():
                if file.is_file():
                    file.unlink()
            temporary.rmdir()


if __name__ == '__main__':
    unittest.main()
