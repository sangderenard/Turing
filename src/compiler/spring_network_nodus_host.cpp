// Persistent host for one native Nodus spring row. Physics remains in the
// registered row and its Turing-compiled law; this process supplies its clock.
#include "tool_api.h"
#include "value_types.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
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
  uint64_t table_shape[1]={6};
  uint64_t table=nodus_tensor_create(NODUS_DTYPE_F64,1,table_shape);
  const double seconds=.01;
  if(!config||!dt||!control||!table||
     nodus_tensor_write(config,0,data.data(),Count*sizeof(double))!=Count*sizeof(double)||
     nodus_tensor_write(dt,0,&seconds,sizeof(seconds))!=sizeof(seconds)||
     !invoke(tool,config))return 4;
  int requested=0; // 0 running, 1 frozen, 2 quit
  HANDLE commands=GetStdHandle(STD_INPUT_HANDLE);
  std::string line;
  bool frozen=false,okay=true;
  std::vector<double> step_us;
  auto next=std::chrono::steady_clock::now();
  long long tick_us=10000;
  if(const char* configured=std::getenv("NODUS_FABRIC_TICK_US")){
    try{
      const auto value=std::stoll(configured);
      if(value>=0&&value<=100000)tick_us=value;
    }catch(const std::exception&){}
  }
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
        else if(line.rfind("substeps ",0)==0){
          try{
            const int count=std::stoi(line.substr(9));
            if(count>=1&&count<=128&&line=="substeps "+std::to_string(count)){
              const double value[2]={3,static_cast<double>(count)};
              okay=nodus_tensor_write(control,0,value,sizeof(value))==sizeof(value)&&invoke(tool,control);
            }
          }catch(const std::exception&){okay=false;}
        }
        else if(line.rfind("target ",0)==0){
          try{
            const double target=std::stod(line.substr(7));
            if(std::isfinite(target)&&target>=0&&target<=8){
              const double value[2]={4,target};
              okay=nodus_tensor_write(control,0,value,sizeof(value))==sizeof(value)&&invoke(tool,control);
            }
          }catch(const std::exception&){okay=false;}
        }
        else if(line.rfind("floor ",0)==0){
          try{
            const double floor=std::stod(line.substr(6));
            if(std::isfinite(floor)){
              const double value[2]={5,floor};
              okay=nodus_tensor_write(control,0,value,sizeof(value))==sizeof(value)&&invoke(tool,control);
            }
          }catch(const std::exception&){okay=false;}
        }
        else if(line.rfind("table ",0)==0){
          std::istringstream source(line.substr(6));
          double bounds[6]={2};
          if((source>>bounds[1]>>bounds[2]>>bounds[3]>>bounds[4]>>bounds[5])&&
             nodus_tensor_write(table,0,bounds,sizeof(bounds))==sizeof(bounds))
            okay=invoke(tool,table);
          else okay=false;
        }
        line.clear();
      }else if(ch!='\r'&&line.size()<128)line.push_back(ch);
    }
    if(requested==2)break;
    const bool want_frozen=requested==1;
    if(want_frozen!=frozen){
      double value[2]={1,want_frozen?1.0:0.0};
      okay=nodus_tensor_write(control,0,value,sizeof(value))==sizeof(value)&&invoke(tool,control);
      frozen=want_frozen;
    }
    if(!frozen&&okay){
      const auto begin=std::chrono::steady_clock::now();
      okay=invoke(tool,dt);
      if(okay)step_us.push_back(std::chrono::duration<double,std::micro>(
        std::chrono::steady_clock::now()-begin).count());
    }
    if(tick_us){
      next+=std::chrono::microseconds(tick_us);
      std::this_thread::sleep_until(next);
      if(std::chrono::steady_clock::now()-next>std::chrono::milliseconds(100))
        next=std::chrono::steady_clock::now();
    }
  }
  nodus_tensor_destroy(table);nodus_tensor_destroy(control);nodus_tensor_destroy(dt);nodus_tensor_destroy(config);
  tool->shutdown();destroy(tool);FreeLibrary(module);
  if(!step_us.empty()){
    std::sort(step_us.begin(),step_us.end());
    double sum=0;for(double value:step_us)sum+=value;
    std::cout<<"spring_host_steps="<<step_us.size()
      <<" mean_us="<<sum/step_us.size()
      <<" median_us="<<step_us[step_us.size()/2]
      <<" p95_us="<<step_us[static_cast<size_t>(.95*(step_us.size()-1))]
      <<" max_us="<<step_us.back()<<'\n';
  }
  return okay?0:5;
}
