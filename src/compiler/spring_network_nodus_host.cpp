// Persistent host for one native Nodus spring row. Physics remains in the
// registered row and its Turing-compiled law; this process supplies its clock.
#include "tool_api.h"
#include "value_types.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include <windows.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using nodus::tensors::AbstractTensorHandle;
static constexpr size_t Count=8*256*38;

static bool invoke(ITool* tool,uint64_t input){
  RawStackFrame frame{};
  if(!raw_stack_init_frame(frame,4096))return false;
  AbstractTensorHandle source{input},output{};
  bool okay=raw_stack_push_abstract_tensor(frame,source);
  ToolStackContext context{};context.stack.raw=&frame;
  if(okay)tool->execute_stack(context);
  okay=okay&&context.succeeded&&raw_stack_pop_abstract_tensor(frame,output);
  if(output.id)nodus_tensor_destroy(output.id);
  raw_stack_free_mask(frame);if(frame.backend)gp_mem_backend_release(frame.backend);
  return okay;
}

int main(int argc,char** argv){
  if(argc!=5)return 2;
  _putenv_s("NODUS_FABRIC_UDP_HOST",argv[3]);
  _putenv_s("NODUS_FABRIC_UDP_PORT",argv[4]);
  std::ifstream input(argv[2],std::ios::binary);
  std::vector<double> data(Count);
  if(!input.read(reinterpret_cast<char*>(data.data()),Count*sizeof(double))||input.peek()!=EOF)return 2;
  HMODULE module=LoadLibraryA(argv[1]);if(!module)return 3;
  auto create=reinterpret_cast<ITool*(*)()>(GetProcAddress(module,"create_tool"));
  auto destroy=reinterpret_cast<void(*)(ITool*)>(GetProcAddress(module,"destroy_tool"));
  if(!create||!destroy)return 3;
  ITool* tool=create();tool->initialize(ToolInitContext{});
  uint64_t shape[3]={8,256,38};
  uint64_t config=nodus_tensor_create(NODUS_DTYPE_F64,3,shape);
  uint64_t dt=nodus_tensor_create(NODUS_DTYPE_F64,0,nullptr);
  uint64_t control_shape[1]={2};
  uint64_t control=nodus_tensor_create(NODUS_DTYPE_F64,1,control_shape);
  const double seconds=.01;
  if(!config||!dt||!control||
     nodus_tensor_write(config,0,data.data(),Count*sizeof(double))!=Count*sizeof(double)||
     nodus_tensor_write(dt,0,&seconds,sizeof(seconds))!=sizeof(seconds)||
     !invoke(tool,config))return 4;
  int requested=0; // 0 running, 1 frozen, 2 quit
  HANDLE commands=GetStdHandle(STD_INPUT_HANDLE);
  std::string line;
  bool frozen=false,okay=true;
  auto next=std::chrono::steady_clock::now();
  while(requested!=2&&okay){
    DWORD available=0;
    if(!PeekNamedPipe(commands,nullptr,0,nullptr,&available,nullptr))requested=2;
    while(available--&&requested!=2){
      char ch=0;DWORD read=0;
      if(!ReadFile(commands,&ch,1,&read,nullptr)||read!=1){requested=2;break;}
      if(ch=='\n'){
        if(line=="pause")requested=1;
        else if(line=="resume")requested=0;
        else if(line=="quit")requested=2;
        line.clear();
      }else if(ch!='\r'&&line.size()<32)line.push_back(ch);
    }
    if(requested==2)break;
    const bool want_frozen=requested==1;
    if(want_frozen!=frozen){
      double value[2]={1,want_frozen?1.0:0.0};
      okay=nodus_tensor_write(control,0,value,sizeof(value))==sizeof(value)&&invoke(tool,control);
      frozen=want_frozen;
    }
    if(!frozen&&okay)okay=invoke(tool,dt);
    next+=std::chrono::milliseconds(10);
    std::this_thread::sleep_until(next);
    if(std::chrono::steady_clock::now()-next>std::chrono::milliseconds(100))
      next=std::chrono::steady_clock::now();
  }
  nodus_tensor_destroy(control);nodus_tensor_destroy(dt);nodus_tensor_destroy(config);
  tool->shutdown();destroy(tool);FreeLibrary(module);
  return okay?0:5;
}
