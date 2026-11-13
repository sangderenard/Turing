// Standalone acceptance for the user-space compiled Nodus spring row.
// argv[1] is the generated row DLL; run with nodus_tensor_core.dll on PATH.
#include "tool_api.h"
#include "value_types.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#pragma comment(lib,"ws2_32.lib")

using nodus::tensors::AbstractTensorHandle;
static constexpr int Nets=8, Nodes=256, Axes=3, Fields=38;

static bool invoke(ITool* tool,uint64_t input,std::vector<double>& positions){
  RawStackFrame frame{};
  if(!raw_stack_init_frame(frame,4096))return false;
  AbstractTensorHandle source{input},output{};
  bool okay=raw_stack_push_abstract_tensor(frame,source);
  ToolStackContext context{};context.stack.raw=&frame;
  if(okay)tool->execute_stack(context);
  okay=okay&&context.succeeded&&raw_stack_pop_abstract_tensor(frame,output);
  if(okay){positions.resize(Nets*Nodes*Axes);okay=nodus_tensor_read(output.id,0,positions.data(),positions.size()*sizeof(double))==
    static_cast<int64_t>(positions.size()*sizeof(double));}
  if(output.id)nodus_tensor_destroy(output.id);
  raw_stack_free_mask(frame);if(frame.backend)gp_mem_backend_release(frame.backend);
  return okay;
}

int main(int argc,char** argv){
  if(argc!=2)return 2;
  _putenv_s("NODUS_FABRIC_UDP_HOST","127.0.0.1");
  _putenv_s("NODUS_FABRIC_UDP_PORT","48769");
  WSADATA wsa{};if(WSAStartup(MAKEWORD(2,2),&wsa))return 2;
  SOCKET receiver=socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);
  sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(48769);address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
  DWORD timeout=1000;setsockopt(receiver,SOL_SOCKET,SO_RCVTIMEO,reinterpret_cast<char*>(&timeout),sizeof(timeout));
  if(bind(receiver,reinterpret_cast<sockaddr*>(&address),sizeof(address))){std::cerr<<"UDP bind failed\n";return 2;}
  HMODULE module=LoadLibraryA(argv[1]);if(!module){std::cerr<<"Plugin load failed "<<GetLastError()<<"\n";return 2;}
  auto create=reinterpret_cast<ITool*(*)()>(GetProcAddress(module,"create_tool"));
  auto destroy=reinterpret_cast<void(*)(ITool*)>(GetProcAddress(module,"destroy_tool"));
  if(!create||!destroy)return 2;
  ITool* tool=create();tool->initialize(ToolInitContext{});
  uint64_t shape[3]={Nets,Nodes,Fields};
  uint64_t config=nodus_tensor_create(NODUS_DTYPE_F64,3,shape);
  std::vector<double> data(Nets*Nodes*Fields);
  auto cell=[&](int node,int field)->double&{return data[node*Fields+field];};
  cell(0,37)=42;
  for(int node=0;node<3;node++){
    cell(node,0)=node;cell(node,6)=1;cell(node,11)=1;cell(node,12)=10;
    cell(node,8)=node;cell(node,9)=0;cell(node,10)=0;
  }
  cell(0,7)=cell(2,7)=1;
  cell(1,1)=1;cell(1,17)=0;cell(1,18)=2;
  cell(1,21)=-1;cell(1,24)=1;
  cell(1,33)=cell(1,34)=1;
  cell(3,0)=7;
  if(!config||nodus_tensor_write(config,0,data.data(),data.size()*sizeof(double))!=
      static_cast<int64_t>(data.size()*sizeof(double)))return 2;
  std::vector<double> output;
  bool configured=invoke(tool,config,output);
  std::cerr<<"configured="<<configured<<" output_y="<<(output.size()>4?output[4]:-999)<<"\n";
  bool okay=configured&&std::abs(output[4]-1)<1e-12;
  uint64_t dt=nodus_tensor_create(NODUS_DTYPE_F64,0,nullptr);
  double seconds=.1;
  okay=okay&&dt&&nodus_tensor_write(dt,0,&seconds,sizeof(seconds))==sizeof(seconds);
  uint64_t control_shape[1]={2};
  uint64_t control=nodus_tensor_create(NODUS_DTYPE_F64,1,control_shape);
  double command[2]={1,1};
  okay=okay&&control&&nodus_tensor_write(control,0,command,sizeof(command))==sizeof(command);
  okay=okay&&invoke(tool,control,output)&&std::abs(output[4]-1)<1e-12;
  okay=okay&&invoke(tool,dt,output)&&std::abs(output[4]-1)<1e-12;
  command[1]=0;
  okay=okay&&nodus_tensor_write(control,0,command,sizeof(command))==sizeof(command);
  okay=okay&&invoke(tool,control,output)&&std::abs(output[4]-1)<1e-12;
  bool stepped=invoke(tool,dt,output);
  std::cerr<<"stepped="<<stepped<<" output_y="<<(output.size()>4?output[4]:-999)<<"\n";
  okay=okay&&stepped&&std::abs(output[4]-.8)<1e-12&&
    std::abs(output[0])<1e-12&&std::abs(output[6]-2)<1e-12&&
    std::abs(output[9]-7)<1e-12;
  auto receive=[&](uint64_t wanted_frame,float wanted_y,uint16_t frozen){
    unsigned char packet[800]{};const int got=recv(receiver,reinterpret_cast<char*>(packet),sizeof(packet),0);
    uint32_t magic=0,node=0;uint64_t frame=0,identity=0;uint16_t count=0,state=0;float y=0;
    if(got>=80){std::memcpy(&magic,packet,4);std::memcpy(&frame,packet+8,8);
      std::memcpy(&identity,packet+16,8);std::memcpy(&count,packet+28,2);
      std::memcpy(&state,packet+30,2);std::memcpy(&node,packet+48,4);std::memcpy(&y,packet+56,4);}
    return got==80&&magic==0x4241464e&&frame==wanted_frame&&identity==42&&
      count==3&&state==frozen&&node==1&&std::abs(y-wanted_y)<1e-5f;
  };
  okay=okay&&receive(1,1.0f,0)&&receive(2,1.0f,1)&&
    receive(3,1.0f,0)&&receive(4,.8f,0);
  uint64_t table_shape[1]={6};
  uint64_t table=nodus_tensor_create(NODUS_DTYPE_F64,1,table_shape);
  double bounds[6]={2,.75,0,2,-1,1};
  okay=okay&&table&&nodus_tensor_write(table,0,bounds,sizeof(bounds))==sizeof(bounds);
  okay=okay&&invoke(tool,table,output)&&std::abs(output[4]-.8)<1e-12;
  okay=okay&&invoke(tool,dt,output)&&std::abs(output[4]-.75)<1e-12;
  okay=okay&&receive(5,.75f,0);
  // One row call with two hot substeps must equal two one-step calls from
  // the same configured initial state, including neighbor and contact reads.
  ITool* reference=create();reference->initialize(ToolInitContext{});
  std::vector<double> expected,actual;
  okay=okay&&invoke(tool,config,actual)&&invoke(reference,config,expected);
  okay=okay&&invoke(tool,table,actual)&&invoke(reference,table,expected);
  command[0]=3;command[1]=2;
  okay=okay&&nodus_tensor_write(control,0,command,sizeof(command))==sizeof(command);
  okay=okay&&invoke(tool,control,actual);
  seconds=.1;
  okay=okay&&nodus_tensor_write(dt,0,&seconds,sizeof(seconds))==sizeof(seconds);
  okay=okay&&invoke(tool,dt,actual);
  seconds=.05;
  okay=okay&&nodus_tensor_write(dt,0,&seconds,sizeof(seconds))==sizeof(seconds);
  okay=okay&&invoke(reference,dt,expected)&&invoke(reference,dt,expected);
  if(okay)for(size_t i=0;i<actual.size();i++)
    if(std::abs(actual[i]-expected[i])>1e-11){okay=false;break;}
  command[0]=4;command[1]=2;
  okay=okay&&nodus_tensor_write(control,0,command,sizeof(command))==sizeof(command);
  okay=okay&&invoke(tool,control,actual)&&invoke(tool,dt,actual);
  cell(1,0)=.5;cell(1,7)=2;cell(1,8)=.5;cell(1,9)=1;
  okay=okay&&nodus_tensor_write(config,0,data.data(),data.size()*sizeof(double))==
    static_cast<int64_t>(data.size()*sizeof(double));
  okay=okay&&invoke(tool,config,actual);
  seconds=.1;
  okay=okay&&nodus_tensor_write(dt,0,&seconds,sizeof(seconds))==sizeof(seconds);
  const bool ring_step=invoke(tool,dt,actual);
  std::cerr<<"ring_step="<<ring_step<<" x="<<(actual.size()>5?actual[3]:-999)
           <<" y="<<(actual.size()>5?actual[4]:-999)<<" z="<<(actual.size()>5?actual[5]:-999)<<"\n";
  okay=okay&&ring_step&&actual[3]>.5&&
    std::abs(actual[4]-1)<1e-12&&std::abs(actual[5])<1e-12;
  reference->shutdown();destroy(reference);
  tool->shutdown();destroy(tool);FreeLibrary(module);
  nodus_tensor_destroy(config);nodus_tensor_destroy(dt);nodus_tensor_destroy(control);nodus_tensor_destroy(table);
  closesocket(receiver);WSACleanup();
  std::cout<<(okay?"spring row + UDP: PASS":"spring row + UDP: FAIL")<<"\n";
  return okay?0:1;
}
