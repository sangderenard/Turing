// Runs the real compiled Nodus row and records its UDP fabric packets as CSV.
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
#include <fstream>
#include <iostream>
#include <vector>
#pragma comment(lib,"ws2_32.lib")

using nodus::tensors::AbstractTensorHandle;
static constexpr int Nets=8, Nodes=256, Fields=38, Width=9, Height=9;

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
  if(argc!=3)return 2;
  _putenv_s("NODUS_FABRIC_UDP_HOST","127.0.0.1");
  _putenv_s("NODUS_FABRIC_UDP_PORT","48770");
  WSADATA wsa{};if(WSAStartup(MAKEWORD(2,2),&wsa))return 2;
  SOCKET receiver=socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);
  sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(48770);
  address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
  DWORD timeout=1000;setsockopt(receiver,SOL_SOCKET,SO_RCVTIMEO,reinterpret_cast<char*>(&timeout),sizeof(timeout));
  if(bind(receiver,reinterpret_cast<sockaddr*>(&address),sizeof(address)))return 2;
  HMODULE module=LoadLibraryA(argv[1]);if(!module)return 2;
  auto create=reinterpret_cast<ITool*(*)()>(GetProcAddress(module,"create_tool"));
  auto destroy=reinterpret_cast<void(*)(ITool*)>(GetProcAddress(module,"destroy_tool"));
  if(!create||!destroy)return 2;
  ITool* tool=create();tool->initialize(ToolInitContext{});
  uint64_t shape[3]={Nets,Nodes,Fields};
  uint64_t config=nodus_tensor_create(NODUS_DTYPE_F64,3,shape);
  std::vector<double> data(Nets*Nodes*Fields);
  auto field=[&](int node,int column)->double&{return data[node*Fields+column];};
  auto node_at=[](int x,int y){return y*Width+x;};
  field(0,37)=20261112;
  for(int y=0;y<Height;y++)for(int x=0;x<Width;x++){
    const int node=node_at(x,y);
    field(node,0)=0.18*x;field(node,1)=-0.18*y;
    field(node,2)=0.08*std::sin(0.7*x)*std::sin(0.5*y);
    field(node,6)=1;field(node,7)=(y==0&&(x==0||x==Width-1))?1:0;
    field(node,8)=field(node,0);field(node,9)=field(node,1);field(node,10)=field(node,2);
    field(node,11)=1;field(node,12)=600;field(node,13)=0.35;field(node,15)=-9.8;
    const int dx[4]={-1,1,0,0},dy[4]={0,0,-1,1};
    for(int edge=0;edge<4;edge++){
      const int nx=x+dx[edge],ny=y+dy[edge];
      if(nx<0||nx>=Width||ny<0||ny>=Height)continue;
      field(node,17+edge)=node_at(nx,ny);
      field(node,21+edge*3)=0.18*dx[edge];
      field(node,22+edge*3)= -0.18*dy[edge];
      field(node,33+edge)=1;
    }
  }
  if(!config||nodus_tensor_write(config,0,data.data(),data.size()*sizeof(double))!=
    static_cast<int64_t>(data.size()*sizeof(double)))return 2;
  uint64_t dt=nodus_tensor_create(NODUS_DTYPE_F64,0,nullptr);
  double seconds=.01;
  if(!dt||nodus_tensor_write(dt,0,&seconds,sizeof(seconds))!=sizeof(seconds))return 2;
  std::ofstream csv(argv[2]);if(!csv)return 2;
  csv<<"frame,node,x,y,z\n";
  bool okay=true;
  for(int frame=0;frame<=100&&okay;frame++){
    okay=invoke(tool,frame?dt:config);
    int got_nodes=0;
    for(int chunk=0;chunk<2&&okay;chunk++){
      unsigned char packet[800]{};
      const int got=recv(receiver,reinterpret_cast<char*>(packet),sizeof(packet),0);
      if(got<32){okay=false;break;}
      uint32_t magic=0;uint64_t sequence=0,identity=0;uint16_t part=0,chunks=0,count=0;
      std::memcpy(&magic,packet,4);std::memcpy(&sequence,packet+8,8);
      std::memcpy(&identity,packet+16,8);std::memcpy(&part,packet+24,2);
      std::memcpy(&chunks,packet+26,2);std::memcpy(&count,packet+28,2);
      if(magic!=0x4241464e||sequence!=static_cast<uint64_t>(frame+1)||
         identity!=20261112||part!=chunk||chunks!=2||got!=32+16*count){okay=false;break;}
      for(int i=0;i<count;i++){
        const auto* pair=packet+32+16*i;
        uint32_t node=0;float xyz[3]{};
        std::memcpy(&node,pair,4);std::memcpy(xyz,pair+4,12);
        csv<<frame<<','<<node<<','<<xyz[0]<<','<<xyz[1]<<','<<xyz[2]<<'\n';
        got_nodes++;
      }
    }
    okay=okay&&got_nodes==Width*Height;
  }
  tool->shutdown();destroy(tool);FreeLibrary(module);
  nodus_tensor_destroy(config);nodus_tensor_destroy(dt);
  closesocket(receiver);WSACleanup();
  std::cout<<(okay?"cloth UDP demo: 101 complete frames, 81 nodes each":"cloth UDP demo failed")<<'\n';
  return okay?0:1;
}
