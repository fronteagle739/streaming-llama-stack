/* Normal Studio ds4_binary entry point. No model or CUDA work here: launch
 * the existing Python coordinator under its independent memory guardian. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <wchar.h>

static void fail(const char*why){fprintf(stderr,"DS4COMBO_LAUNCH_REFUSED %s win32=%lu\n",why,GetLastError());ExitProcess(72);}
static void put(wchar_t*out,size_t*used,wchar_t value){if(*used>=32766)fail("command line exceeds Windows bound");out[(*used)++]=value;out[*used]=0;}
/* The standard Windows CRT/list2cmdline backslash-and-quote convention. */
static void argument(wchar_t*out,size_t*used,const wchar_t*value){
    if(*used)put(out,used,L' ');put(out,used,L'"');
    while(*value){size_t slashes=0;while(*value==L'\\'){slashes++;value++;}
        if(!*value){for(size_t i=0;i<slashes*2;i++)put(out,used,L'\\');break;}
        if(*value==L'"'){for(size_t i=0;i<slashes*2+1;i++)put(out,used,L'\\');put(out,used,*value++);}
        else{for(size_t i=0;i<slashes;i++)put(out,used,L'\\');put(out,used,*value++);}
    }
    put(out,used,L'"');
}
int wmain(int argc,wchar_t**argv){
    wchar_t root[32768],path[32768],python[4096],line[32768]={0};
    DWORD n=GetModuleFileNameW(NULL,root,32768);if(!n||n>=32768)fail("module path");
    wchar_t*separator=wcsrchr(root,L'\\');if(!separator)fail("absolute module directory required");*separator=0;
    if(_snwprintf_s(path,32768,_TRUNCATE,L"%ls\\python-path.txt",root)<0)fail("Python path file bound");
    HANDLE file=CreateFileW(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(file==INVALID_HANDLE_VALUE)fail("installed Python path file");
    char bytes[4096];DWORD got=0;
    if(!ReadFile(file,bytes,4095,&got,NULL)||!got||GetFileSize(file,NULL)!=got)fail("Python path file size");CloseHandle(file);
    while(got&&(bytes[got-1]=='\r'||bytes[got-1]=='\n'))got--;bytes[got]=0;
    if(!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,bytes,-1,python,4096))fail("Python path UTF8");
    size_t used=0;argument(line,&used,python);argument(line,&used,L"-B");argument(line,&used,L"-m");argument(line,&used,L"slstack.combo_proxy");
    argument(line,&used,L"--config");if(_snwprintf_s(path,32768,_TRUNCATE,L"%ls\\combo.json",root)<0)fail("config path bound");argument(line,&used,path);
    argument(line,&used,L"--native-argv");for(int i=1;i<argc;i++)argument(line,&used,argv[i]);
    STARTUPINFOW startup={0};PROCESS_INFORMATION child={0};startup.cb=sizeof(startup);startup.dwFlags=STARTF_USESTDHANDLES|STARTF_USESHOWWINDOW;
    startup.wShowWindow=SW_HIDE;startup.hStdInput=GetStdHandle(STD_INPUT_HANDLE);startup.hStdOutput=GetStdHandle(STD_OUTPUT_HANDLE);startup.hStdError=GetStdHandle(STD_ERROR_HANDLE);
    if(!CreateProcessW(python,line,NULL,NULL,TRUE,CREATE_NO_WINDOW,NULL,root,&startup,&child))fail("guarded coordinator process");
    CloseHandle(child.hThread);if(WaitForSingleObject(child.hProcess,INFINITE)!=WAIT_OBJECT_0)fail("coordinator wait");
    DWORD code=72;if(!GetExitCodeProcess(child.hProcess,&code))fail("coordinator exit status");CloseHandle(child.hProcess);return(int)code;
}
