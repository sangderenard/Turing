// Saturated native Nodus row trial: three physical Vulkan workers and one
// LLVM worker borrow disjoint tensors from one state-library tool row. Each
// worker has its own result FIFO; one collator writes the shared sink FIFO.
#include "tensor_state_library_tool.h"
#include "gpu_fabric_dispatch_tool.h"
#include "cpu_fabric_dispatch_tool.h"
#include "tool_registry.h"
#include "edge_tensor_fifo.h"
#include "fabric_resident_offer.h"
#include "gpu_shader_job_tool.h"
#include "nodus_headless_abi.h"

#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr std::uint64_t nodes_per_fabric=256ull*1024,lanes_per_fabric=nodes_per_fabric*3;
using Request=std::array<std::uint64_t,12>;
std::array<std::uint64_t,32> call_library(nodus::gpu::TensorStateLibraryTool& tool,
                                           const Request& request,std::uint64_t payload){
    const std::uint64_t shape[1]={request.size()};
    auto command=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    if(!command||nodus_tensor_write(command,0,request.data(),sizeof(request))!=sizeof(request))
        return {UINT64_MAX};
    RawStackFrame frame{};
    if(!raw_stack_init_frame(frame,4096))return {UINT64_MAX};
    raw_stack_push_abstract_tensor(frame,{command});
    raw_stack_push_abstract_tensor(frame,{payload});
    ToolStackContext context{};context.stack.raw=&frame;
    tool.execute_stack(context);
    nodus::tensors::AbstractTensorHandle output{};
    std::array<std::uint64_t,32> result{UINT64_MAX};
    if(context.succeeded&&raw_stack_pop_abstract_tensor(frame,output)){
        nodus_tensor_read(output.id,0,result.data(),sizeof(result));
        nodus_tensor_destroy(output.id);
    }
    nodus_tensor_destroy(command);
    raw_stack_free_mask(frame);if(frame.backend)gp_mem_backend_release(frame.backend);
    return result;
}
double feed_value(int feed,std::uint64_t lane,std::uint64_t width,
                  std::uint64_t height){
    const auto nodes=width*height;
    const auto node=(lane/3)%nodes;
    const auto axis=lane%3,x=node%width,y=node/width;
    const double position=axis==0?x*.01:axis==1?(height-y)*.01:0;
    const double velocity=axis==0?((lane/(nodes*3))%31)*.0001:0;
    switch(feed){
    case 0:return position;
    case 1:case 2:return 0;
    case 3:return .05;
    case 4:return .001;
    case 5:return x>0?1:0;
    case 6:return x+1<width?1:0;
    case 7:return y>0?1:0;
    case 8:return y+1<height?1:0;
    case 9:return axis==1?-9.8:0;
    case 10:return 1;
    case 11:return position-(axis==0&&x>0?.01:0);
    case 12:return position+(axis==0&&x+1<width?.01:0);
    case 13:return position+(axis==1&&y>0?.01:0);
    case 14:return position-(axis==1&&y+1<height?.01:0);
    case 15:return 1;
    case 16:return y==0?1:0;
    case 17:return position;
    case 18:return axis==0?-.01:0;
    case 19:return axis==0?.01:0;
    case 20:return axis==1?.01:0;
    case 21:return axis==1?-.01:0;
    case 22:return 10;
    case 23:return velocity;
    default:return 0;
    }
}
struct Job {std::string tool;std::uint64_t base=0,lanes=0,width=256,height=1024;};
std::uint64_t steady_ns(){
    return static_cast<std::uint64_t>(std::chrono::duration_cast<
        std::chrono::nanoseconds>(std::chrono::steady_clock::now()
            .time_since_epoch()).count());
}
struct MarketRound {
    std::unique_ptr<nodus::contracts::ContractNegotiator> market;
    nodus::contracts::Loan accepted{};
    std::uint64_t selected_base=0;
    std::uint64_t recent_cold_ns=20000000,recent_warm_ns=20000000;
    std::uint64_t cold_base=0,warm_base=0,expires_cold=0,expires_warm=0;
    bool completed=false,sent=false,committed=false;
};
double median(std::vector<double>& values){
    std::sort(values.begin(),values.end());
    return values[values.size()/2];
}
bool put_job(nodus::gpu::TensorStateLibraryTool& library,std::uint64_t card,
             const Job& job){
    std::vector<double> values(job.lanes);
    for(int feed=0;feed<27;feed++){
        std::uint64_t payload=0;
        if(feed==26){
            const std::uint64_t shape[1]={2};
            payload=nodus_tensor_create(NODUS_DTYPE_U32,1,shape);
            const std::uint32_t control[2]={static_cast<std::uint32_t>(job.lanes),
                                            static_cast<std::uint32_t>(job.width*3)};
            nodus_tensor_write(payload,0,control,sizeof(control));
        }else{
            const std::uint64_t shape[1]={job.lanes};
            payload=nodus_tensor_create(NODUS_DTYPE_F64,1,shape);
            if(!payload)return false;
            if(feed<24)for(std::uint64_t lane=0;lane<job.lanes;lane++)
                values[lane]=feed_value(feed,lane,job.width,job.height);
            else std::fill(values.begin(),values.end(),0);
            if(nodus_tensor_write(payload,0,values.data(),values.size()*8)!=
               static_cast<int64_t>(values.size()*8))return false;
        }
        Request request{};request[0]=2;request[1]=card;request[3]=job.base+feed;
        const auto receipt=call_library(library,request,payload);
        nodus_tensor_destroy(payload);
        if(receipt[0]){
            std::cerr<<"put refused job="<<job.tool<<" feed="<<feed
                     <<" status="<<receipt[0]<<'\n';
            return false;
        }
    }
    return true;
}
bool put_ray_job(nodus::gpu::TensorStateLibraryTool& library,
                 std::uint64_t card,const Job& job){
    const auto bits=[](float value){return std::bit_cast<std::uint32_t>(value);};
    const auto n=static_cast<std::size_t>(job.lanes);
    std::array<std::vector<std::uint32_t>,9> fields;
    fields[0].resize(n*86);
    fields[1].resize(n*93);
    fields[2].resize(24);
    fields[3].resize(10);
    fields[4].resize(1);
    fields[5].resize(16);
    fields[6].resize(32*12);
    fields[7].resize(2);
    fields[8].resize(8);
    for(std::size_t ray=0;ray<n;ray++){
        auto* intent=fields[0].data()+ray*86;
        intent[0]=bits(-.25f);intent[1]=bits(-.25f);
        intent[2]=bits(-1.0f);
        intent[3]=bits(.01f);intent[4]=bits(.01f);
        intent[5]=bits(1.0f);
        intent[7]=static_cast<std::uint32_t>(-1);
        intent[11]=1;intent[22]=bits(1.0f);
    }
    fields[2][8]=static_cast<std::uint32_t>(n);
    fields[3]={bits(-1.0f),bits(-1.0f),bits(-.1f),
               bits(1.0f),bits(1.0f),bits(.1f),
               static_cast<std::uint32_t>(-1),
               static_cast<std::uint32_t>(-1),0,1};
    fields[5]={bits(-.5f),bits(-.5f),bits(0),
               bits(1),bits(0),bits(0),
               bits(0),bits(1),bits(0),
               bits(0),bits(0),bits(1),
               0,0,static_cast<std::uint32_t>(-1),
               static_cast<std::uint32_t>(-1)};
    fields[8]={static_cast<std::uint32_t>(n),1,1,0,1,0,0,0};
    for(std::size_t i=0;i<fields.size();i++){
        const std::uint64_t shape[1]={fields[i].size()};
        const auto tensor=nodus_tensor_create(NODUS_DTYPE_U32,1,shape);
        if(!tensor||nodus_tensor_write(tensor,0,fields[i].data(),
             fields[i].size()*sizeof(std::uint32_t))!=
             static_cast<std::int64_t>(fields[i].size()*sizeof(std::uint32_t)))
            return false;
        Request request{};request[0]=2;request[1]=card;request[3]=job.base+i;
        const auto result=call_library(library,request,tensor);
        nodus_tensor_destroy(tensor);
        if(result[0])return false;
    }
    return true;
}

int canvas_trial(nodus::gpu::TensorStateLibraryTool& library,
                 std::uint64_t card,const std::array<Job,4>& jobs,
                 const std::array<Job,3>& alternate,
                 const std::array<Job,3>& ray_jobs,int rounds,int cpu_workers,
                 bool alternate_states,bool overload,int overload_device,
                 bool deadline,bool light_load,bool cpu_only,bool ray_mode){
    auto* canvas=nodus_headless_create();
    if(!canvas)return 20;
    std::vector<Job> scheduled;
    if(overload_device>=0){
        scheduled={jobs[overload_device],alternate[overload_device]};
    }else if(overload){
        if(!cpu_only)for(int i=0;i<3;i++){
            scheduled.push_back(jobs[i]);
            if(!light_load)scheduled.push_back(alternate[i]);
            if(ray_mode)scheduled.push_back(ray_jobs[i]);
        }
        for(int i=0;i<cpu_workers;i++){
            auto cpu=jobs[3];
            if(i)cpu.base=8000+static_cast<std::uint64_t>(i-1)*1000;
            scheduled.push_back(cpu);
        }
    }else scheduled.assign(jobs.begin(),jobs.end());
    const auto worker_count=static_cast<int>(scheduled.size());
    const std::string label=deadline?"deadline":overload_device>=0?
        "gpu"+std::to_string(overload_device):
        cpu_only?"cpu-only":light_load?"mixed-light":
        ray_mode?"mixed-ray":
        overload?"overload":alternate_states?"cold":"warm";
    const auto tensor_type=nodus_headless_abstract_tensor_type_id();
    NodusHeadlessFifoConfig fifo{};
    fifo.token_bytes=sizeof(std::uint64_t);fifo.slots=16;
    fifo.type_id=tensor_type;fifo.layout=2;fifo.dtype=8;
    std::vector<std::string> input_names(worker_count),output_names(worker_count);
    for(int i=0;i<worker_count;i++){
        GP_CanvasModuleDesc module{};
        module.x=i*100;module.y=0;module.w=module.h=64;
        const int index=nodus_headless_add_module(canvas,&module);
        const GP_CanvasModuleRowDesc rows[3]={{0,0,1,0,nullptr},
            {1,0,1,1,scheduled[i].tool.c_str()},{2,0,1,0,nullptr}};
        if(index<0||!nodus_headless_set_module_rows(canvas,index,rows,3)||
           !nodus_headless_set_module_stack_contract(canvas,index,1)){
            nodus_headless_destroy(canvas);return 21;
        }
        input_names[i]="fabric.in."+std::to_string(i);
        output_names[i]="fabric.out."+std::to_string(i);
        if(!nodus_headless_declare_input_port_ex(canvas,input_names[i].c_str(),
                index,0,&fifo)||
           !nodus_headless_declare_output_port_ex(canvas,output_names[i].c_str(),
                index,1,&fifo,0)){
            nodus_headless_destroy(canvas);return 22;
        }
    }
    if(!nodus_headless_set_scheduler_mode(canvas,1)||
       !nodus_headless_set_persistent_workers(canvas,1)||
       !nodus_headless_set_parallel_worker_limit(canvas,
            worker_count)){
        nodus_headless_destroy(canvas);return 23;
    }
    std::vector<std::vector<double>> lease_ms(worker_count),
        dispatch_ms(worker_count),total_ms(worker_count);
    std::vector<std::vector<std::uint64_t>> steps(worker_count);
    std::array<std::uint64_t,3> recent_primary{20000000,20000000,20000000};
    std::array<std::uint64_t,3> recent_alternate{20000000,20000000,20000000};
    std::array<std::uint64_t,3> last_base{jobs[0].base,jobs[1].base,jobs[2].base};
    if(deadline)for(int i=0;i<3;i++)for(int feed=0;feed<27;feed++){
        auto warm=library.borrow_gpu(card,jobs[i].base+feed,i,false);
        if(!warm){nodus_headless_destroy(canvas);return 29;}
    }
    const auto wall_start=std::chrono::steady_clock::now();
    double steady_seconds=0;
    std::vector<double> round_ms;
    for(int round=0;round<rounds;round++){
        const auto round_start=std::chrono::steady_clock::now();
        std::array<std::unique_ptr<MarketRound>,3> markets;
        if(deadline)for(int i=0;i<3;i++){
            auto rig=std::make_unique<MarketRound>();
            auto* state=rig.get();
            state->warm_base=last_base[i];
            state->cold_base=last_base[i]==jobs[i].base?
                alternate[i].base:jobs[i].base;
            state->recent_cold_ns=state->cold_base==jobs[i].base?
                recent_primary[i]:recent_alternate[i];
            state->recent_warm_ns=state->warm_base==jobs[i].base?
                recent_primary[i]:recent_alternate[i];
            const auto now=steady_ns();
            state->expires_cold=now+state->recent_cold_ns/2;
            state->expires_warm=now+10*state->recent_warm_ns;
            auto offer=nodus::gpu::resident_fabric_offer(library,
                [i](std::uint64_t peer)->std::optional<std::size_t>{
                    if(peer==static_cast<std::uint64_t>(100+i))return i;
                    return {};
                },[state,card](std::uint64_t base)->
                    std::optional<nodus::gpu::FabricStateRef>{
                    if(base==state->cold_base)return nodus::gpu::FabricStateRef{
                        card,base,27,state->expires_cold,state->recent_cold_ns};
                    if(base==state->warm_base)return nodus::gpu::FabricStateRef{
                        card,base,27,state->expires_warm,state->recent_warm_ns};
                    return {};
                });
            state->market=std::make_unique<nodus::contracts::ContractNegotiator>(
                2,std::move(offer),10);
            nodus::contracts::Producer engine{};engine.id=900+i;
            engine.request=[](nodus::contracts::Contract&){return false;};
            engine.route=[state](const nodus::contracts::Result& result){
                state->committed=result.payload==state->selected_base;
                return state->committed;
            };
            nodus::contracts::Peer host{};host.id=100+i;
            host.exchange=[state](const nodus::contracts::Loan* offered){
                nodus::contracts::Exchange reply{};
                if(offered){state->accepted=*offered;
                    state->selected_base=offered->work.payload;
                    reply.accepted_generation=offered->generation;}
                if(state->completed&&!state->sent){
                    reply.has_result=true;
                    reply.result={state->accepted,state->selected_base};
                    state->sent=true;
                }
                return reply;
            };
            const std::uint64_t first_id=1000+round*10+i*2;
            if(!state->market->add_engine(engine)||
               !state->market->add_host(host)||
               !state->market->submit({first_id,engine.id,1,state->cold_base})||
               !state->market->submit({first_id+1,engine.id,1,state->warm_base})){
                nodus_headless_destroy(canvas);return 30;
            }
            state->market->market_turn();
            if(state->selected_base!=state->cold_base){
                nodus_headless_destroy(canvas);return 31;
            }
            markets[i]=std::move(rig);
        }
        std::vector<std::uint64_t> command(worker_count);
        for(int i=0;i<worker_count;i++){
            const auto& job=deadline&&i<3?
                (markets[i]->selected_base==jobs[i].base?jobs[i]:alternate[i]):
                !overload&&i<3&&alternate_states&&round%2?
                alternate[i]:scheduled[i];
            const std::uint64_t shape[1]={4};
            command[i]=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
            const std::uint64_t words[4]={card,job.base,job.lanes,
                                          static_cast<std::uint64_t>(round+1)};
            if(!command[i]||nodus_tensor_write(command[i],0,words,sizeof(words))!=
                    sizeof(words)||
               !nodus_headless_push_token(canvas,input_names[i].c_str(),
                   &command[i],sizeof(command[i]))){
                nodus_headless_destroy(canvas);return 24;
            }
        }
        std::vector<bool> received(worker_count,false);
        int completed=0;
        for(int tick=0;tick<24&&completed<worker_count;tick++){
            if(!nodus_headless_step(canvas,1.0/60.0)){
                nodus_headless_destroy(canvas);return 25;
            }
            for(int i=0;i<worker_count;i++){
                if(received[i])continue;
                std::uint64_t token=0;std::size_t written=0;
                if(!nodus_headless_pull_token(canvas,output_names[i].c_str(),
                        &token,sizeof(token),&written))continue;
                std::array<std::uint64_t,9> event{};
                const bool cpu=scheduled[i].tool=="nodus.fabric.dispatch.cpu";
                const auto bytes=9*sizeof(std::uint64_t);
                const auto& expected=deadline&&i<3?
                    (markets[i]->selected_base==jobs[i].base?jobs[i]:alternate[i]):
                    !overload&&i<3&&alternate_states&&round%2?
                    alternate[i]:scheduled[i];
                if(written!=sizeof(token)||
                   nodus_tensor_read(token,0,event.data(),bytes)!=
                       static_cast<std::int64_t>(bytes)||
                   event[0]||event[5]!=expected.base){
                    std::cerr<<"canvas receipt failed worker="<<i
                             <<" round="<<round<<" status="<<event[0]<<'\n';
                    nodus_headless_destroy(canvas);return 26;
                }
                lease_ms[i].push_back(event[6]*1e-6);
                dispatch_ms[i].push_back(event[3]*1e-6);
                total_ms[i].push_back(event[7]*1e-6);
                steps[i].push_back(event[8]);
                if(deadline&&i<3){
                    auto& rig=*markets[i];
                    rig.completed=true;
                    rig.market->market_turn();rig.market->worker_turn();
                    if(!rig.committed){nodus_headless_destroy(canvas);return 32;}
                    auto& average=expected.base==jobs[i].base?
                        recent_primary[i]:recent_alternate[i];
                    average=(average*3+event[7])/4;
                    last_base[i]=expected.base;
                }
                std::cout<<"canvas_sample mode="
                         <<label
                         <<" worker="<<scheduled[i].tool
                         <<" round="<<round+1<<" base="<<expected.base
                         <<" steps="<<event[8]
                         <<" lease_ms="<<event[6]*1e-6
                         <<" dispatch_ms="<<event[3]*1e-6
                         <<" total_ms="<<event[7]*1e-6<<'\n';
                nodus_tensor_destroy(token);
                received[i]=true;++completed;
            }
        }
        for(const auto id:command)nodus_tensor_destroy(id);
        if(completed!=worker_count){nodus_headless_destroy(canvas);return 27;}
        const auto round_time=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-round_start).count();
        round_ms.push_back(round_time*1000);
        if(round)steady_seconds+=round_time;
        std::cout<<"canvas_round mode="
                 <<label
                 <<" round="<<round+1<<" wall_ms="<<round_time*1000<<'\n';
    }
    const double elapsed=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-wall_start).count();
    for(int i=0;i<worker_count;i++){
        std::sort(steps[i].begin(),steps[i].end());
        std::cout<<"canvas_mode="<<label
                 <<" worker="<<scheduled[i].tool<<" base="<<scheduled[i].base
                 <<" rounds="<<rounds
                 <<" median_steps="<<steps[i][steps[i].size()/2]
                 <<" median_lease_ms="<<median(lease_ms[i])
                 <<" median_dispatch_ms="<<median(dispatch_ms[i])
                 <<" median_total_ms="<<median(total_ms[i])<<'\n';
    }
    std::cout<<"canvas_mode="<<label
             <<" results="<<rounds*worker_count<<" elapsed_s="<<elapsed
             <<" median_round_ms="<<median(round_ms)
             <<" steady_jobs_per_s="<<(rounds>1?
                 (rounds-1)*worker_count/steady_seconds:0)
             <<" cpu_worker_limit="<<(overload&&overload_device<0?cpu_workers:1)
             <<" pool_worker_limit="<<worker_count
             <<'\n';
    for(int i=0;i<3;i++){
        if(overload_device>=0&&i!=overload_device)continue;
        auto state=library.borrow_gpu(card,jobs[i].base+17,i,false);
        nodus::kernels::VulkanBackendStatus status{};std::string error;
        if(state&&nodus::kernels::query_vulkan_backend_status(
                state.handle(),&status,&error))
            std::cout<<"canvas_device="<<i<<" name="<<status.device.name
                     <<" queue_family="<<status.device.queue_family
                     <<" available_queues="<<status.device.available_compute_queues
                     <<" active_queues="<<status.active_compute_queues<<'\n';
        if(const auto metrics=library.device_performance(i))
            std::cout<<"canvas_metrics device="<<i
                     <<" samples="<<metrics->samples
                     <<" average_lease_ms="<<metrics->average_lease_ns*1e-6
                     <<" average_dispatch_ms="<<metrics->average_dispatch_ns*1e-6
                     <<" average_total_ms="<<metrics->average_total_ns*1e-6
                     <<" cache_resident_bytes="<<metrics->cache_resident_bytes
                     <<" cache_budget_bytes="<<metrics->cache_budget_bytes<<'\n';
    }
    std::vector<Job> evolved=scheduled;
    if(deadline||alternate_states)
        evolved.insert(evolved.end(),alternate.begin(),alternate.end());
    for(const auto& job:evolved){
        if(job.tool.rfind("nodus.shader.ray_bvh.vulkan.",0)==0){
            const auto device=static_cast<std::size_t>(job.tool.back()-'0');
            auto state=library.borrow_gpu(card,job.base+2,device,false);
            const auto* gpu=state?gp_mem_backend_get_vtable(state.handle()):nullptr;
            std::array<std::uint32_t,2> counts{};
            if(!gpu||!gpu->copy_from_backend(state.handle(),0,
                    counts.data(),sizeof(counts))||counts[0]!=job.lanes||counts[1]){
                std::cerr<<"ray hit mismatch device="<<device
                         <<" hits="<<counts[0]<<" misses="<<counts[1]
                         <<" expected="<<job.lanes<<'\n';
                nodus_headless_destroy(canvas);return 33;
            }
            std::cout<<"canvas_ray_hit device="<<device
                     <<" base="<<job.base<<" hits="<<counts[0]<<'\n';
            continue;
        }
        if(job.tool=="nodus.fabric.dispatch.cpu"){
            auto state=library.borrow_cpu(card,job.base+17,false);
            const auto lane=(job.width+1)*3+1;
            const double observed=state&&state.bytes()>(lane+1)*sizeof(double)?
                static_cast<const double*>(state.data())[lane]:NAN;
            if(!std::isfinite(observed)||
               std::abs(observed-feed_value(17,lane,job.width,job.height))<1e-8){
                nodus_headless_destroy(canvas);return 28;
            }
            std::cout<<"canvas_evolved device=cpu base="<<job.base
                     <<" sample_y="<<observed<<'\n';
            continue;
        }
        const auto device=static_cast<std::size_t>(job.tool.back()-'0');
        auto state=library.borrow_gpu(card,job.base+17,device,false);
        const auto* gpu=state?gp_mem_backend_get_vtable(state.handle()):nullptr;
        const auto lane=(job.width+1)*3+1;
        double observed=0;
        if(!gpu||!gpu->copy_from_backend(state.handle(),lane*sizeof(double),
                                         &observed,sizeof(observed))||
           !std::isfinite(observed)||
           std::abs(observed-feed_value(17,lane,job.width,job.height))<1e-8){
            nodus_headless_destroy(canvas);return 28;
        }
        std::cout<<"canvas_evolved device="<<device<<" base="<<job.base
                 <<" sample_y="<<observed<<'\n';
    }
    nodus_headless_destroy(canvas);
    return 0;
}
}

int main(int argc,char** argv){
    if(argc<7||argc>10){std::cerr<<"usage: profile_tensor_dispatch_network SPV LLVM_DLL rounds local_size max_steps target_completion_ns [mode] [cpu_workers_1_to_6] [ray_spv]\n";return 2;}
    const std::string mode=argc>=8?argv[7]:"direct";
    const int cpu_workers=argc>=9?std::stoi(argv[8]):1;
    const bool ray_mode=mode=="canvas-mixed-ray";
    if(ray_mode!=(argc==10))return 2;
    if(cpu_workers<1||cpu_workers>6||
       (mode!="canvas-overload"&&mode!="canvas-mixed-ray"&&
        mode!="canvas-mixed-light"&&
        mode!="canvas-cpu-only"&&cpu_workers!=1))return 2;
    if(mode!="direct"&&mode!="canvas-warm"&&mode!="canvas-cold"&&
       mode!="canvas-deadline"&&
       mode!="canvas-overload"&&mode!="canvas-mixed-ray"&&
       mode!="canvas-mixed-light"&&
       mode!="canvas-cpu-only"&&mode!="canvas-gpu0"&&
       mode!="canvas-gpu1"&&mode!="canvas-gpu2")return 2;
    const int overload_device=mode.rfind("canvas-gpu",0)==0?mode.back()-'0':-1;
    const bool overload=mode=="canvas-overload"||ray_mode||
        mode=="canvas-mixed-light"||mode=="canvas-cpu-only"||overload_device>=0;
    const int rounds=std::stoi(argv[3]);
    const auto local_size=static_cast<std::uint32_t>(std::stoul(argv[4]));
    if(rounds<1||rounds>32)return 2;
    std::ifstream file(argv[1],std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(file),{}};
    if(bytes.empty()||bytes.size()%4)return 2;
    std::vector<std::uint32_t> shader(bytes.size()/4);
    std::memcpy(shader.data(),bytes.data(),bytes.size());
    nodus::gpu::TensorStateLibraryTool library(6ull*1024*1024*1024,
        {800ull*1024*1024,120ull*1024*1024,1536ull*1024*1024});
    library.initialize({});
    if(library.device_count()!=3)return 3;
    Request status_request{};status_request[0]=4;
    const auto status=call_library(library,status_request,0);
    constexpr std::uint64_t margin=1024ull*1024;
    if(status[6]<=margin||status[11]<=margin||status[16]<=margin)return 4;
    const auto gtx_quota=std::min<std::uint64_t>(
        800ull*1024*1024,status[6]-margin);
    const auto amd_quota=std::min<std::uint64_t>(
        120ull*1024*1024,status[11]-margin);
    const auto rtx_quota=std::min<std::uint64_t>(
        1536ull*1024*1024,status[16]-margin);
    // The overload comparison uses identical 256x256x3 fabrics and one
    // resident loadout contract on every GPU and on LLVM CPU.
    const auto equal_lanes=256ull*256*3;
    const auto amd_lanes=overload?equal_lanes:256ull*512*3;
    if(amd_quota<amd_lanes*26*8)return 4;
    const std::uint64_t card=7001;
    Request card_request{};card_request[0]=1;card_request[1]=card;
    card_request[2]=901;card_request[4]=6ull*1024*1024*1024;
    card_request[5]=gtx_quota;
    card_request[6]=amd_quota;
    card_request[7]=rtx_quota;
    const auto card_result=call_library(library,card_request,0);
    if(card_result[0]){
        std::cerr<<"card refused status="<<card_result[0]
                 <<" budgets="<<status[6]<<','<<status[11]<<','<<status[16]
                 <<" quotas="<<gtx_quota<<','<<amd_quota<<','<<rtx_quota<<'\n';
        return 5;
    }
    const std::array<Job,4> jobs{{
        {"nodus.fabric.dispatch.vulkan.0",1000,overload?equal_lanes:4*lanes_per_fabric,256,overload?256ull:1024ull},
        {"nodus.fabric.dispatch.vulkan.1",2000,amd_lanes,256,overload?256ull:512ull},
        {"nodus.fabric.dispatch.vulkan.2",3000,overload?equal_lanes:8*lanes_per_fabric,256,overload?256ull:1024ull},
        {"nodus.fabric.dispatch.cpu",4000,overload?equal_lanes:lanes_per_fabric,256,overload?256ull:1024ull}
    }};
    const std::array<Job,3> alternate{{
        {"nodus.fabric.dispatch.vulkan.0",5000,overload?equal_lanes:4*lanes_per_fabric,256,overload?256ull:1024ull},
        {"nodus.fabric.dispatch.vulkan.1",6000,amd_lanes,256,overload?256ull:512ull},
        {"nodus.fabric.dispatch.vulkan.2",7000,overload?equal_lanes:8*lanes_per_fabric,256,overload?256ull:1024ull}
    }};
    const std::array<Job,3> ray_jobs{{
        {"nodus.shader.ray_bvh.vulkan.0",13000,16384,0,0},
        {"nodus.shader.ray_bvh.vulkan.1",14000,16384,0,0},
        {"nodus.shader.ray_bvh.vulkan.2",15000,16384,0,0}
    }};
    for(const auto& job:jobs)if(!put_job(library,card,job))return 6;
    for(const auto& job:alternate)if(!put_job(library,card,job))return 6;
    if(ray_mode)for(const auto& job:ray_jobs)
        if(!put_ray_job(library,card,job))return 6;
    if(overload&&overload_device<0)for(int i=1;i<cpu_workers;i++){
        auto job=jobs[3];job.base=8000+static_cast<std::uint64_t>(i-1)*1000;
        if(!put_job(library,card,job))return 6;
    }
    const nodus::gpu::GpuShaderWorkContract contract{
        static_cast<std::uint32_t>(std::stoul(argv[5])),true,
        std::stoull(argv[6])};
    if(!nodus::gpu::register_gpu_fabric_dispatchers(
           tool_registry_global(),library,shader,local_size,contract)||
       !nodus::gpu::register_cpu_fabric_dispatcher(
           tool_registry_global(),library,argv[2],
           "spring_network_step_batched__tick",
           overload?equal_lanes:lanes_per_fabric,
           overload?contract.steps_per_loadout:1))return 7;
    if(ray_mode){
        std::ifstream ray_file(argv[9],std::ios::binary);
        std::vector<char> ray_bytes{std::istreambuf_iterator<char>(ray_file),{}};
        if(ray_bytes.empty()||ray_bytes.size()%4)return 7;
        nodus::gpu::GpuShaderJobContract ray{};
        ray.id="nodus.shader.ray_bvh";
        ray.program_id=2;
        ray.spirv.resize(ray_bytes.size()/4);
        std::memcpy(ray.spirv.data(),ray_bytes.data(),ray_bytes.size());
        ray.local_size_x=64;ray.maximum_invocations=16384;
        ray.minimum_binding_bytes={16384ull*86*4,16384ull*93*4,24*4,
            10*4,4,16*4,32*12*4,2*4,8*4};
        ray.writable={0,1,1,0,0,0,0,0,0};
        ray.reset_before_dispatch={{2,0,3,0}};
        if(!nodus::gpu::register_gpu_shader_job(
                tool_registry_global(),library,ray))return 7;
    }
    if(mode!="direct"){
        const int result=canvas_trial(library,card,jobs,alternate,ray_jobs,
                                      rounds,cpu_workers,
                                      mode=="canvas-cold",overload,
                                      overload_device,mode=="canvas-deadline",
                                      mode=="canvas-mixed-light",
                                      mode=="canvas-cpu-only",ray_mode);
        if(result)return result;
        for(const auto& job:jobs)for(int feed=0;feed<27;feed++){
            Request request{};request[0]=3;request[1]=card;request[3]=job.base+feed;
            if(call_library(library,request,0)[0])return 10;
        }
        for(const auto& job:alternate)for(int feed=0;feed<27;feed++){
            Request request{};request[0]=3;request[1]=card;request[3]=job.base+feed;
            if(call_library(library,request,0)[0])return 10;
        }
        if(overload&&overload_device<0)for(int i=1;i<cpu_workers;i++)
            for(int feed=0;feed<27;feed++){
                Request request{};request[0]=3;request[1]=card;
                request[3]=8000+static_cast<std::uint64_t>(i-1)*1000+feed;
                if(call_library(library,request,0)[0])return 10;
            }
        if(ray_mode)for(const auto& job:ray_jobs)
            for(int feed=0;feed<9;feed++){
                Request request{};request[0]=3;request[1]=card;
                request[3]=job.base+feed;
                if(call_library(library,request,0)[0])return 10;
            }
        library.shutdown();return 0;
    }
    const auto type=ValueTypeRegistry::global().register_primitive("dispatch_result_token",8);
    std::array<EdgeTensorFifo,4> results;
    EdgeTensorFifo sink;
    for(int i=0;i<4;i++){
        results[i].configure({1},16,0,8,type);
        if(!results[i].subscribe(100+i,true))return 8;
    }
    sink.configure({1},16,0,8,type);
    if(!sink.subscribe(999,true))return 8;
    std::atomic<int> failures{0};
    std::array<std::thread,4> workers;
    const auto started=std::chrono::steady_clock::now();
    for(int i=0;i<4;i++)workers[i]=std::thread([&,i]{
        auto worker=tool_registry_global().create(jobs[i].tool);
        if(!worker){failures++;return;}
        worker->initialize({});
        const std::uint64_t shape[1]={4};
        const auto command=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
        for(int round=0;round<rounds;round++){
            const auto& selected=i<3&&round%2?alternate[i]:jobs[i];
            const std::uint64_t request[4]={card,selected.base,selected.lanes,
                                             static_cast<std::uint64_t>(round+1)};
            nodus_tensor_write(command,0,request,sizeof(request));
            RawStackFrame frame{};
            if(!raw_stack_init_frame(frame,4096)){failures++;break;}
            raw_stack_push_abstract_tensor(frame,{command});
            ToolStackContext context{};context.stack.raw=&frame;
            worker->execute_stack(context);
            nodus::tensors::AbstractTensorHandle output{};
            if(!context.succeeded||!raw_stack_pop_abstract_tensor(frame,output)){
                std::cerr<<"worker execution failed "<<jobs[i].tool
                         <<" round="<<round+1<<'\n';
                failures++;break;
            }
            bool dropped=false;
            if(!results[i].push(1000+i,2000+i,&output.id,8,&dropped)||dropped)
                failures++;
            raw_stack_free_mask(frame);
            if(frame.backend)gp_mem_backend_release(frame.backend);
        }
        nodus_tensor_destroy(command);
        worker->shutdown();
    });
    std::array<std::vector<double>,4> dispatch_ms,lease_ms,total_ms;
    std::array<std::vector<std::uint64_t>,4> step_counts;
    int accepted=0;
    while(accepted<rounds*4&&failures==0){
        bool progress=false;
        for(int i=0;i<4;i++){
            std::uint64_t token=0;std::size_t read=0;
            if(!results[i].pop(100+i,&token,8,read))continue;
            progress=true;
            bool dropped=false;
            if(!sink.push(900,901,&token,8,&dropped)||dropped){failures++;break;}
            std::uint64_t received=0;
            if(!sink.pop(999,&received,8,read)||received!=token){failures++;break;}
            std::array<std::uint64_t,9> event{};
            const auto receipt_bytes=i<3?sizeof(event):sizeof(std::uint64_t)*8;
            if(nodus_tensor_read(token,0,event.data(),receipt_bytes)!=
               static_cast<std::int64_t>(receipt_bytes)||
               event[0]!=0){
                std::cerr<<"receipt failure "<<jobs[i].tool
                         <<" status="<<event[0]<<" generation="<<event[4]<<'\n';
                failures++;break;
            }
            if(i<3&&(event[8]<1||event[8]>contract.steps_per_loadout)){
                std::cerr<<"receipt step count mismatch "<<jobs[i].tool
                         <<" generation="<<event[4]<<" steps="<<event[8]<<'\n';
                failures++;break;
            }
            const auto& expected=i<3&&event[4]%2==0?alternate[i]:jobs[i];
            if(event[5]!=expected.base){
                std::cerr<<"receipt selected wrong state "<<jobs[i].tool<<'\n';
                failures++;break;
            }
            dispatch_ms[i].push_back(event[3]*1e-6);
            step_counts[i].push_back(i<3?event[8]:1);
            lease_ms[i].push_back(event[6]*1e-6);
            total_ms[i].push_back(event[7]*1e-6);
            nodus_tensor_destroy(token);
            ++accepted;
        }
        if(!progress)std::this_thread::yield();
    }
    for(auto& thread:workers)thread.join();
    if(failures||accepted!=rounds*4)return 9;
    const auto seconds=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-started).count();
    for(int i=0;i<4;i++){
        std::sort(dispatch_ms[i].begin(),dispatch_ms[i].end());
        std::sort(step_counts[i].begin(),step_counts[i].end());
        std::sort(lease_ms[i].begin(),lease_ms[i].end());
        std::sort(total_ms[i].begin(),total_ms[i].end());
        std::cout<<jobs[i].tool<<" lanes="<<jobs[i].lanes
                 <<" rounds="<<dispatch_ms[i].size()
                 <<" median_steps_per_loadout="
                 <<step_counts[i][step_counts[i].size()/2]
                 <<" median_lease_ms="<<lease_ms[i][lease_ms[i].size()/2]
                 <<" median_dispatch_ms="<<dispatch_ms[i][dispatch_ms[i].size()/2]
                 <<" median_total_ms="<<total_ms[i][total_ms[i].size()/2]<<'\n';
    }
    std::cout<<"shared_sink_results="<<accepted<<" elapsed_s="<<seconds
             <<" ram_bytes="<<call_library(library,status_request,0)[3]<<'\n';
    const auto end_status=call_library(library,status_request,0);
    for(int i=0;i<3;i++){
        const auto evicted=end_status[8+i*5];
        std::cout<<"device="<<i<<" evicted="<<evicted
                 <<" reused="<<end_status[9+i*5]<<'\n';
        if(rounds>2&&!evicted)return 12;
    }
    for(int i=0;i<3;i++)for(const Job* selected:{&jobs[i],&alternate[i]}){
        auto state=library.borrow_gpu(card,selected->base+17,i,false);
        const auto* gpu=state?gp_mem_backend_get_vtable(state.handle()):nullptr;
        const std::uint64_t lane=(selected->width+1)*3+1;
        double observed=0;
        if(!gpu||!gpu->copy_from_backend(state.handle(),lane*sizeof(double),
                                         &observed,sizeof(observed))||
           !std::isfinite(observed)||
           std::abs(observed-feed_value(17,lane,selected->width,selected->height))<1e-8)
            return 11;
        std::cout<<"evolved_state_device="<<i<<" base="<<selected->base
                 <<" sample_y="<<observed<<'\n';
    }
    for(const auto& job:jobs)for(int feed=0;feed<27;feed++){
        Request request{};request[0]=3;request[1]=card;request[3]=job.base+feed;
        if(call_library(library,request,0)[0])return 10;
    }
    for(const auto& job:alternate)for(int feed=0;feed<27;feed++){
        Request request{};request[0]=3;request[1]=card;request[3]=job.base+feed;
        if(call_library(library,request,0)[0])return 10;
    }
    library.shutdown();
    return 0;
}
