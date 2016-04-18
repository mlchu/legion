/* Copyright 2016 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LOWLEVEL_HSA_H
#define LOWLEVEL_HSA_H

#include "hsa.h"

#include "realm/operation.h"
#include "realm/module.h"
#include "realm/threads.h"
#include "realm/circ_queue.h"
#include "realm/indexspace.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"


namespace Realm {
  namespace HSA {

    class GPU;
    class GPUWorker;
    struct GPUInfo;

    // our interface to the rest of the runtime
    class HSAModule : public Module {
    protected:
      HSAModule(void);
      
    public:
      virtual ~HSAModule(void);

      static Module *create_module(RuntimeImpl *runtime, std::vector<std::string>& cmdline);

      // do any general initialization - this is called after all configuration is
      //  complete
      virtual void initialize(RuntimeImpl *runtime);

      // create any memories provided by this module (default == do nothing)
      //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
      virtual void create_memories(RuntimeImpl *runtime);

      // create any processors provided by the module (default == do nothing)
      //  (each new ProcessorImpl should use a Processor from
      //   RuntimeImpl::next_local_processor_id)
      virtual void create_processors(RuntimeImpl *runtime);

      // create any DMA channels provided by the module (default == do nothing)
      virtual void create_dma_channels(RuntimeImpl *runtime);

      // create any code translators provided by the module (default == do nothing)
      virtual void create_code_translators(RuntimeImpl *runtime);

      // clean up any common resources created by the module - this will be called
      //  after all memories/processors/etc. have been shut down and destroyed
      virtual void cleanup(void);

    public:
      unsigned cfg_num_gpus;
      hsa_agent_t agent;

      // "global" variables live here too
      GPUWorker *shared_worker;
      std::map<GPU *, GPUWorker *> dedicated_workers;
      std::vector<GPUInfo *> gpu_info;
      std::vector<GPU *> gpus;
    };

    REGISTER_REALM_MODULE(HSAModule);

    struct GPUInfo {
      static const size_t MAX_NAME_LEN = 64;
      char name[MAX_NAME_LEN];
    }

    // Forard declaration
    class GPUProcessor;
    class GPUWorker;

    // an interface for receiving completion notification for a GPU operation
    //  (right now, just copies)
    class GPUCompletionNotification {
    public:
      virtual ~GPUCompletionNotification(void) {}

      virtual void request_completed(void) = 0;
    };

    // a class that represents a CUDA stream and work associated with 
    //  it (e.g. queued copies, events in flight)
    // a stream is also associated with a GPUWorker that it will register
    //  with when async work needs doing
    class GPUStream {
    public:
      GPUStream(GPU *_gpu, GPUWorker *_worker);
      ~GPUStream(void);

      GPU *get_gpu(void) const;
      //      CUstream get_stream(void) const;

      // may be called by anybody to enqueue a copy or an event
      //      void add_copy(GPUMemcpy *copy);
      //      void add_fence(GPUWorkFence *fence);
      void add_notification(GPUCompletionNotification *notification);

      // to be called by a worker (that should already have the GPU context
      //   current) - returns true if any work remains
      bool issue_copies(void);
      bool reap_events(void);

    protected:
      //      void add_event(CUevent event, GPUWorkFence *fence, 
      //		     GPUCompletionNotification *notification);

      GPU *gpu;
      GPUWorker *worker;

      GASNetHSL mutex;

      struct PendingEvent {
        //	CUevent event;
        //	GPUWorkFence *fence;
        GPUCompletionNotification* notification;
      };
#ifdef USE_CQ
      Realm::CircularQueue<PendingEvent> pending_events;
#else
      std::deque<PendingEvent> pending_events;
#endif
    };

    // a GPUWorker is responsible for making progress on one or more GPUStreams -
    //  this may be done directly by a GPUProcessor or in a background thread
    //  spawned for the purpose
    class GPUWorker {
    public:
      GPUWorker(void);
      virtual ~GPUWorker(void);

      // adds a stream that has work to be done
      void add_stream(GPUStream *s);

      // processes work on streams, optionally sleeping for work to show up
      // returns true if work remains to be done
      bool process_streams(bool sleep_on_empty);

      void start_background_thread(Realm::CoreReservationSet& crs,
				   size_t stack_size);
      void shutdown_background_thread(void);

    public:
      void thread_main(void);

    protected:
      GASNetHSL lock;
      GASNetCondVar condvar;
      std::set<GPUStream *> active_streams;

      // used by the background thread (if any)
      Realm::CoreReservation *core_rsrv;
      Realm::Thread *worker_thread;
      bool worker_shutdown_requested;
    };

    struct FatBin;
    struct RegisteredVariable;
    struct RegisteredFunction;

    // a GPU object represents our use of a given HSA-capable GPU
    class GPU {
    public:
      GPU(HSAModule *_module);
      ~GPU(void);

      void push_context(void);
      void pop_context(void);

      void register_fat_binary(const FatBin *data);
      void register_variable(const RegisteredVariable *var);
      void register_function(const RegisteredFunction *func);

      void create_processor(RuntimeImpl *runtime, size_t stack_size);
      void create_fb_memory(RuntimeImpl *runtime, size_t size);

      void create_dma_channels(Realm::RuntimeImpl *r);

      GPUStream *switch_to_next_task_stream(void);
      GPUStream *get_current_task_stream(void);

    protected:
      //      CUmodule load_cuda_module(const void *data);

    public:
      HSAModule *module;
      GPUWorker *worker;
      GPUProcessor *proc;

      // which system memories have been registered and can be used for cuMemcpyAsync
      std::set<Memory> pinned_sysmems;

      // which other FBs we have peer access to
      std::set<Memory> peer_fbs;

      std::vector<GPUStream *> task_streams;
      unsigned current_stream;
    };

    // helper to push/pop a GPU's context by scope
    class AutoGPUContext {
    public:
      AutoGPUContext(GPU& _gpu);
      AutoGPUContext(GPU *_gpu);
      ~AutoGPUContext(void);
    protected:
      GPU *gpu;
    };

    class GPUProcessor : public Realm::LocalTaskProcessor {
    public:
      GPUProcessor(GPU *_gpu, Processor _me, Realm::CoreReservationSet& crs,
                   size_t _stack_size);
      virtual ~GPUProcessor(void);

    public:
      virtual void shutdown(void);

      static GPUProcessor *get_current_gpu_proc(void);

      void setup_argument(const void *arg, size_t size, size_t offset);
      void launch(const void *func);

    public:
      GPU *gpu;

      // data needed for kernel launches
      struct LaunchConfig {
        /*
        dim3 grid;
        dim3 block;
        size_t shared;
	LaunchConfig(dim3 _grid, dim3 _block, size_t _shared);
        */
      };
      std::vector<LaunchConfig> launch_configs;
      std::vector<char> kernel_args;

    protected:
      Realm::CoreReservation *core_rsrv;
    };
  }; // namespace LowLevel
}; // namespace LegionRuntime

#endif
