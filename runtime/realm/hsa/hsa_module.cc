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

#include "hsa_module.h"

#include "realm/tasks.h"
#include "realm/logging.h"
#include "realm/cmdline.h"

#include "lowlevel_dma.h"

#include "activemsg.h"
#include "realm/utils.h"

#include <stdio.h>

namespace Realm {
  namespace HSA {

    // dma code is still in old namespace
    typedef LegionRuntime::LowLevel::DmaRequest DmaRequest;
    typedef LegionRuntime::LowLevel::OASVec OASVec;
    typedef LegionRuntime::LowLevel::InstPairCopier InstPairCopier;
    typedef LegionRuntime::LowLevel::MemPairCopier MemPairCopier;
    typedef LegionRuntime::LowLevel::MemPairCopierFactory MemPairCopierFactory;

    Logger log_gpu("gpu");
    Logger log_gpudma("gpudma");

#ifdef EVENT_GRAPH_TRACE
    extern Logger log_event_graph;
#endif
    Logger log_stream("gpustream");


  ////////////////////////////////////////////////////////////////////////
  //
  // class GPUStream

    GPUStream::GPUStream(GPU *_gpu, GPUWorker *_worker)
      : gpu(_gpu), worker(_worker)
    {
      assert(worker != 0);
      log_stream.info() << "HSA stream created for GPU " << gpu;
    }

    GPUStream::~GPUStream(void)
    {
        log_stream.info() << "HSA stream destroyed ";
    }

    GPU *GPUStream::get_gpu(void) const
    {
      return gpu;
    }

#if 0    
    // may be called by anybody to enqueue a copy or an event
    void GPUStream::add_copy(GPUMemcpy *copy)
    {
      bool add_to_worker = false;
      {
	AutoHSLLock al(mutex);

	// remember to add ourselves to the worker if we didn't already have work
	add_to_worker = pending_copies.empty();

	pending_copies.push_back(copy);
      }

      if(add_to_worker)
	worker->add_stream(this);
    }

    void GPUStream::add_fence(GPUWorkFence *fence)
    {
        /*
      CUevent e = gpu->event_pool.get_event();

      CHECK_CU( cuEventRecord(e, stream) );

      log_stream.debug() << "CUDA event " << e << " recorded on stream " << stream << " (GPU " << gpu << ")";

      add_event(e, fence, 0);
        */
    }
#endif
    void GPUStream::add_notification(GPUCompletionNotification *notification)
    {
        /*
      CUevent e = gpu->event_pool.get_event();

      CHECK_CU( cuEventRecord(e, stream) );
      add_event(e, 0, notification);
        */
    }
      /*
    void GPUStream::add_event(CUevent event, GPUWorkFence *fence, 
			      GPUCompletionNotification *notification)
    {
      bool add_to_worker = false;
      {
	AutoHSLLock al(mutex);

	// remember to add ourselves to the worker if we didn't already have work
	add_to_worker = pending_events.empty();

	PendingEvent e;
	e.event = event;
	e.fence = fence;
	e.notification = notification;

	pending_events.push_back(e);
      }

      if(add_to_worker)
	worker->add_stream(this);
    }
      */
    // to be called by a worker (that should already have the GPU context
    //   current) - returns true if any work remains
    bool GPUStream::issue_copies(void)
    {
        /*
      while(true) {
	GPUMemcpy *copy = 0;
	{
	  AutoHSLLock al(mutex);

	  if(pending_copies.empty())
	    return false;  // no work left

	  copy = pending_copies.front();
	  pending_copies.pop_front();
	}

	{
	  AutoGPUContext agc(gpu);
	  copy->execute(this);
	}

	// no backpressure on copies yet - keep going until list is empty
      }
        */
        return false;
    }

    bool GPUStream::reap_events(void)
    {
        /*
      // peek at the first event
      CUevent event;
      bool event_valid = false;
      {
	AutoHSLLock al(mutex);

	if(pending_events.empty())
	  return false;  // no work left

	event = pending_events.front().event;
	event_valid = true;
      }

      // we'll keep looking at events until we find one that hasn't triggered
      while(event_valid) {
	CUresult res = cuEventQuery(event);

	if(res == CUDA_ERROR_NOT_READY)
	  return true; // oldest event hasn't triggered - check again later

	// no other kind of error is expected
	assert(res == CUDA_SUCCESS);

	log_stream.debug() << "CUDA event " << event << " triggered on stream " << stream << " (GPU " << gpu << ")";

	// give event back to GPU for reuse
	gpu->event_pool.return_event(event);

	// this event has triggered, so figure out the fence/notification to trigger
	//  and also peek at the next event
	GPUWorkFence *fence = 0;
	GPUCompletionNotification *notification = 0;

	{
	  AutoHSLLock al(mutex);

	  const PendingEvent &e = pending_events.front();
	  assert(e.event == event);
	  fence = e.fence;
	  notification = e.notification;
	  pending_events.pop_front();

	  if(pending_events.empty())
	    event_valid = false;
	  else
	    event = pending_events.front().event;
	}

	if(fence)
	  fence->mark_finished();

	if(notification)
	  notification->request_completed();
      }
        */
      // if we get all the way to here, we're (temporarily, at least) out of work
      return false;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUTaskScheduler<T>

    // we want to subclass the scheduler to replace the execute_task method, but we also want to
    //  allow the use of user or kernel threads, so we apply a bit of template magic (which only works
    //  because the constructors for the KernelThreadTaskScheduler and UserThreadTaskScheduler classes
    //  have the same prototypes)

    template <typename T>
    class GPUTaskScheduler : public T {
    public:
      GPUTaskScheduler(Processor _proc, Realm::CoreReservation& _core_rsrv,
		       GPUProcessor *_gpu_proc);

      virtual ~GPUTaskScheduler(void);

    protected:
      virtual bool execute_task(Task *task);

      // might also need to override the thread-switching methods to keep TLS up to date

      GPUProcessor *gpu_proc;
    };

    template <typename T>
    GPUTaskScheduler<T>::GPUTaskScheduler(Processor _proc,
					  Realm::CoreReservation& _core_rsrv,
					  GPUProcessor *_gpu_proc)
      : T(_proc, _core_rsrv), gpu_proc(_gpu_proc)
    {
      // nothing else
    }

    template <typename T>
    GPUTaskScheduler<T>::~GPUTaskScheduler(void)
    {
    }

    namespace ThreadLocal {
      static __thread GPUProcessor *current_gpu_proc = 0;
    };

    // this flag will be set on the first call into any of the hijack code in
    //  cudart_hijack.cc
    //  an application is linked with -lcudart, we will NOT be hijacking the
    //  application's calls, and the cuda module needs to know that)
    /*extern*/ bool cudart_hijack_active = false;

    // used in GPUTaskScheduler<T>::execute_task below
      //    static bool already_issued_hijack_warning = false;

    template <typename T>
    bool GPUTaskScheduler<T>::execute_task(Task *task)
    {
        printf("in GPUTaskScheduler!\n");
      // use TLS to make sure that the task can find the current GPU processor when it makes
      //  CUDA RT calls
      // TODO: either eliminate these asserts or do TLS swapping when using user threads
      assert(ThreadLocal::current_gpu_proc == 0);
      ThreadLocal::current_gpu_proc = gpu_proc;

      // push the CUDA context for this GPU onto this thread
      gpu_proc->gpu->push_context();

#if 0
      // bump the current stream
      // TODO: sanity-check whether this even works right when GPU tasks suspend
      GPUStream *s = gpu_proc->gpu->switch_to_next_task_stream();

      // we'll use a "work fence" to track when the kernels launched by this task actually
      //  finish - this must be added to the task _BEFORE_ we execute
      GPUWorkFence *fence = new GPUWorkFence(task);
      task->add_async_work_item(fence);
#endif
      bool ok = T::execute_task(task);

      // now enqueue the fence on the local stream
      //      fence->enqueue_on_stream(s);

      // pop the CUDA context for this GPU back off
      gpu_proc->gpu->pop_context();

      assert(ThreadLocal::current_gpu_proc == gpu_proc);
      ThreadLocal::current_gpu_proc = 0;
      printf("done with execute_task\n");

      return ok;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUProcessor

    GPUProcessor::GPUProcessor(GPU *_gpu, Processor _me, Realm::CoreReservationSet& crs,
                               size_t _stack_size)
      : LocalTaskProcessor(_me, Processor::TOC_PROC)
      , gpu(_gpu)
    {
      Realm::CoreReservationParameters params;
      params.set_num_cores(1);
      params.set_alu_usage(params.CORE_USAGE_SHARED);
      params.set_fpu_usage(params.CORE_USAGE_SHARED);
      params.set_ldst_usage(params.CORE_USAGE_SHARED);
      params.set_max_stack_size(_stack_size);

      std::string name = stringbuilder() << "GPU proc " << _me;

      core_rsrv = new Realm::CoreReservation(name, crs, params);

#ifdef REALM_USE_USER_THREADS_FOR_GPU
      Realm::UserThreadTaskScheduler *sched = new GPUTaskScheduler<Realm::UserThreadTaskScheduler>(me, *core_rsrv, this);
      // no config settings we want to tweak yet
#else
      Realm::KernelThreadTaskScheduler *sched = new GPUTaskScheduler<Realm::KernelThreadTaskScheduler>(me, *core_rsrv, this);
      // no config settings we want to tweak yet
#endif
      set_scheduler(sched);
    }

    GPUProcessor::~GPUProcessor(void)
    {
      delete core_rsrv;
    }

    GPUStream *GPU::get_current_task_stream(void)
    {
      return task_streams[current_stream];
    }

    GPUStream *GPU::switch_to_next_task_stream(void)
    {
      current_stream++;
      if(current_stream >= task_streams.size())
	current_stream = 0;
      return task_streams[current_stream];
    }

    void GPUProcessor::shutdown(void)
    {
      log_gpu.info("shutting down");

      // shut down threads/scheduler
      LocalTaskProcessor::shutdown();

      // synchronize the device so we can flush any printf buffers - do
      //  this after shutting down the threads so that we know all work is done
      {
	AutoGPUContext agc(gpu);
      }
    }

    GPUWorker::GPUWorker(void)
      : condvar(lock)
      , core_rsrv(0), worker_thread(0), worker_shutdown_requested(false)
    {}

    GPUWorker::~GPUWorker(void)
    {
      // shutdown should have already been called
      assert(worker_thread == 0);
    }

    void GPUWorker::start_background_thread(Realm::CoreReservationSet &crs,
					    size_t stack_size)
    {
      core_rsrv = new Realm::CoreReservation("GPU worker thread", crs,
					     Realm::CoreReservationParameters());

      Realm::ThreadLaunchParameters tlp;

      worker_thread = Realm::Thread::create_kernel_thread<GPUWorker,
							  &GPUWorker::thread_main>(this,
										   tlp,
										   *core_rsrv,
										   0);
    }

    void GPUWorker::shutdown_background_thread(void)
    {
      {
	AutoHSLLock al(lock);
	worker_shutdown_requested = true;
	condvar.broadcast();
      }

      worker_thread->join();
      delete worker_thread;
      worker_thread = 0;

      delete core_rsrv;
      core_rsrv = 0;
    }

    void GPUWorker::add_stream(GPUStream *stream)
    {
      AutoHSLLock al(lock);

      // if the stream is already in the set, nothing to do
      if(active_streams.count(stream) > 0)
	return;

      active_streams.insert(stream);

      condvar.broadcast();
    }

    bool GPUWorker::process_streams(bool sleep_on_empty)
    {
      // we start by grabbing the list of active streams, replacing it with an
      //  empty list - this way we don't have to hold the lock the whole time
      // for any stream that we leave work on, we'll add it back in
      std::set<GPUStream *> streams;
      {
	AutoHSLLock al(lock);

	while(active_streams.empty()) {
	  if(!sleep_on_empty || worker_shutdown_requested) return false;
	  condvar.wait();
	}

	streams.swap(active_streams);
      }

      bool any_work_left = false;
      for(std::set<GPUStream *>::const_iterator it = streams.begin();
	  it != streams.end();
	  it++) {
	GPUStream *s = *it;
	bool stream_work_left = false;

	if(s->issue_copies())
	  stream_work_left = true;

	if(s->reap_events())
	  stream_work_left = true;

	if(stream_work_left) {
	  add_stream(s);
	  any_work_left = true;
	}
      }

      return any_work_left;
    }

    void GPUWorker::thread_main(void)
    {
      // TODO: consider busy-waiting in some cases to reduce latency?
      while(!worker_shutdown_requested) {
	bool work_left = process_streams(true);

	// if there was work left, yield our thread for now to avoid a tight spin loop
	// TODO: enqueue a callback so we can go to sleep and wake up sooner than a kernel
	//  timeslice?
	if(work_left)
	  Realm::Thread::yield();
      }
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class BlockingCompletionNotification

    class BlockingCompletionNotification : public GPUCompletionNotification {
    public:
      BlockingCompletionNotification(void);
      virtual ~BlockingCompletionNotification(void);

      virtual void request_completed(void);

      virtual void wait(void);

    public:
      GASNetHSL mutex;
      GASNetCondVar cv;
      bool completed;
    };

    BlockingCompletionNotification::BlockingCompletionNotification(void)
      : cv(mutex)
      , completed(false)
    {}

    BlockingCompletionNotification::~BlockingCompletionNotification(void)
    {}

    void BlockingCompletionNotification::request_completed(void)
    {
      AutoHSLLock a(mutex);

      assert(!completed);
      completed = true;
      cv.broadcast();
    }

    void BlockingCompletionNotification::wait(void)
    {
      AutoHSLLock a(mutex);

      while(!completed)
	cv.wait();
    }
	

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPU

    // Helper methods for emulating the cuda runtime
    /*static*/ GPUProcessor* GPUProcessor::get_current_gpu_proc(void)
    {
      return ThreadLocal::current_gpu_proc;
    }


    void GPUProcessor::setup_argument(const void *arg,
				      size_t size, size_t offset)
    {
      size_t required = offset + size;

      if(required > kernel_args.size())
	kernel_args.resize(required);

      memcpy(&kernel_args[offset], arg, size);
    }

    void GPUProcessor::launch(const void *func)
    {
        printf("in GPUProcessor::launch\n");
#if 0
      // make sure we have a launch config
      assert(!launch_configs.empty());
      LaunchConfig &config = launch_configs.back();

      // Find our function
      CUfunction f = gpu->lookup_function(func);

      size_t arg_size = kernel_args.size();
      void *extra[] = { 
        CU_LAUNCH_PARAM_BUFFER_POINTER, &kernel_args[0],
        CU_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        CU_LAUNCH_PARAM_END
      };

      CUstream raw_stream = gpu->get_current_task_stream()->get_stream();
      log_stream.debug() << "kernel " << func << " added to stream " << raw_stream;

      // Launch the kernel on our stream dammit!
      CHECK_CU( cuLaunchKernel(f, 
			       config.grid.x, config.grid.y, config.grid.z,
                               config.block.x, config.block.y, config.block.z,
                               config.shared,
			       raw_stream,
			       NULL, extra) );

      // pop the config we just used
      launch_configs.pop_back();

      // clear out the kernel args
      kernel_args.clear();
#endif
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPU

    GPU::GPU(HSAModule *_module)
      : module(_module)
    {
#if 0
      // create a CUDA context for our device - automatically becomes current
      CHECK_CU( cuCtxCreate(&context, 
			    CU_CTX_MAP_HOST | CU_CTX_SCHED_BLOCKING_SYNC,
			    info->device) );
#endif

    }

    GPU::~GPU(void)
    {
      push_context();

      while(!task_streams.empty()) {
	delete task_streams.back();
	task_streams.pop_back();
      }

    }

    void GPU::push_context(void)
    {
        //      CHECK_CU( cuCtxPushCurrent(context) );
    }

    void GPU::pop_context(void)
    {
      // the context we pop had better be ours...
#if 0
      CUcontext popped;
      CHECK_CU( cuCtxPopCurrent(&popped) );
      assert(popped == context);
#endif
    }

    void GPU::create_processor(RuntimeImpl *runtime, size_t stack_size)
    {
      printf("GPU: create_processor\n");
      Processor p = runtime->next_local_processor_id();
      printf("%d\n", (int) p.id);
      proc = new GPUProcessor(this, p,
			      runtime->core_reservation_set(),
			      stack_size);
      runtime->add_processor(proc);

      MemoryImpl *sysmem;
      std::vector<MemoryImpl *>& local_mems = runtime->nodes[gasnet_mynode()].memories;
      for(std::vector<MemoryImpl *>::iterator it = local_mems.begin();
          it != local_mems.end();
          it++) {
          sysmem = *it;
          if((*it)->kind == MemoryImpl::MKIND_SYSMEM)
              break;
          printf("here: %d\n", (*it)->kind);
      }
      Machine::ProcessorMemoryAffinity pma;
      pma.p = p;
      pma.m = sysmem->me;;
      pma.bandwidth = 100;
      pma.latency = 1;
      runtime->add_proc_mem_affinity(pma);
#if 0
      // this processor is able to access its own FB and the ZC mem (if any)
      Machine::ProcessorMemoryAffinity pma;
      pma.p = p;
      pma.m = fbmem->me;
      pma.bandwidth = 200;  // "big"
      pma.latency = 5;      // "ok"
      runtime->add_proc_mem_affinity(pma);

      if(module->zcmem) {
	pma.m = module->zcmem->me;
	pma.bandwidth = 20; // "medium"
	pma.latency = 200;  // "bad"
	runtime->add_proc_mem_affinity(pma);
      }
#endif
    }

    void GPU::register_fat_binary(const FatBin *fatbin)
    {
#if 0
      AutoGPUContext agc(this);

      log_gpu.info() << "registering fat binary " << fatbin << " with GPU " << this;

      // have we see this one already?
      if(device_modules.count(fatbin) > 0) {
	log_gpu.warning() << "duplicate registration of fat binary data " << fatbin;
	return;
      }

      if(fatbin->data != 0) {
	// binary data to be loaded with cuModuleLoad(Ex)
	CUmodule module = load_cuda_module(fatbin->data);
	device_modules[fatbin] = module;
	return;
      }

      assert(0);
#endif
    }
    
    void GPU::register_variable(const RegisteredVariable *var)
    {
#if 0
      AutoGPUContext agc(this);

      log_gpu.info() << "registering variable " << var->device_name << " (" << var->host_var << ") with GPU " << this;

      // have we seen it already?
      if(device_variables.count(var->host_var) > 0) {
	log_gpu.warning() << "duplicate registration of variable " << var->device_name;
	return;
      }

      // get the module it lives in
      std::map<const FatBin *, CUmodule>::const_iterator it = device_modules.find(var->fat_bin);
      assert(it != device_modules.end());
      CUmodule module = it->second;

      CUdeviceptr ptr;
      size_t size;
      CHECK_CU( cuModuleGetGlobal(&ptr, &size, module, var->device_name) );
      device_variables[var->host_var] = ptr;
#endif
    }
    
    void GPU::register_function(const RegisteredFunction *func)
    {
        printf("in register function\n");
#if 0
      AutoGPUContext agc(this);

      log_gpu.info() << "registering function " << func->device_fun << " (" << func->host_fun << ") with GPU " << this;

      // have we seen it already?
      if(device_functions.count(func->host_fun) > 0) {
	log_gpu.warning() << "duplicate registration of function " << func->device_fun;
	return;
      }

      // get the module it lives in
      std::map<const FatBin *, CUmodule>::const_iterator it = device_modules.find(func->fat_bin);
      assert(it != device_modules.end());
      CUmodule module = it->second;

      CUfunction f;
      CHECK_CU( cuModuleGetFunction(&f, module, func->device_fun) );
      device_functions[func->host_fun] = f;
#endif
    }

#if 0
   CUfunction GPU::lookup_function(const void *func)
    {
      std::map<const void *, CUfunction>::iterator finder = device_functions.find(func);
      assert(finder != device_functions.end());
      return finder->second;
    }

    CUdeviceptr GPU::lookup_variable(const void *var)
    {
      std::map<const void *, CUdeviceptr>::iterator finder = device_variables.find(var);
      assert(finder != device_variables.end());
      return finder->second;
    }

    CUmodule GPU::load_cuda_module(const void *data)
    {
      const unsigned num_options = 4;
      CUjit_option jit_options[num_options];
      void*        option_vals[num_options];
      const size_t buffer_size = 16384;
      char* log_info_buffer = (char*)malloc(buffer_size);
      char* log_error_buffer = (char*)malloc(buffer_size);
      jit_options[0] = CU_JIT_INFO_LOG_BUFFER;
      jit_options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
      jit_options[2] = CU_JIT_ERROR_LOG_BUFFER;
      jit_options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
      option_vals[0] = log_info_buffer;
      option_vals[1] = (void*)buffer_size;
      option_vals[2] = log_error_buffer;
      option_vals[3] = (void*)buffer_size;
      CUmodule module;
      CUresult result = cuModuleLoadDataEx(&module, data, num_options, 
                                           jit_options, option_vals); 
      if (result != CUDA_SUCCESS)
      {
#ifdef __MACH__
        if (result == CUDA_ERROR_OPERATING_SYSTEM) {
          log_gpu.error("ERROR: Device side asserts are not supported by the "
                              "CUDA driver for MAC OSX, see NVBugs 1628896.");
        }
#endif
        if (result == CUDA_ERROR_NO_BINARY_FOR_GPU) {
          log_gpu.error("ERROR: The binary was compiled for the wrong GPU "
                              "architecture. Update the 'GPU_ARCH' flag at the top "
                              "of runtime/runtime.mk to match your current GPU "
                              "architecture.");
        }
        log_gpu.error("Failed to load CUDA module! Error log: %s", 
                log_error_buffer);
#if CUDA_VERSION >= 6050
        const char *name, *str;
        CHECK_CU( cuGetErrorName(result, &name) );
        CHECK_CU( cuGetErrorString(result, &str) );
        fprintf(stderr,"CU: cuModuleLoadDataEx = %d (%s): %s\n",
                result, name, str);
#else
        fprintf(stderr,"CU: cuModuleLoadDataEx = %d\n", result);
#endif
        assert(0);
      }
      else
        log_gpu.info("Loaded CUDA Module. JIT Output: %s", log_info_buffer);
      free(log_info_buffer);
      free(log_error_buffer);
      return module;
    }
#endif
    ////////////////////////////////////////////////////////////////////////
    //
    // class AutoGPUContext

    AutoGPUContext::AutoGPUContext(GPU& _gpu)
      : gpu(&_gpu)
    {
      gpu->push_context();
    }

    AutoGPUContext::AutoGPUContext(GPU *_gpu)
      : gpu(_gpu)
    {
      gpu->push_context();
    }

    AutoGPUContext::~AutoGPUContext(void)
    {
      gpu->pop_context();
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class HSAModule

    // our interface to the rest of the runtime

    HSAModule::HSAModule(void)
      : Module("hsa")
      , cfg_num_gpus(0)
      , shared_worker(0)
    {
        printf("HSAModule constructor\n");
    }
      
    HSAModule::~HSAModule(void)
    {}

    /*
     * Determines if the given agent is of type HSA_DEVICE_TYPE_GPU
     * and sets the value of data to the agent handle if it is.
     */
    static hsa_status_t get_gpu_agent(hsa_agent_t agent, void *data) {
      hsa_status_t status;
      hsa_device_type_t device_type;
      status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
      if (HSA_STATUS_SUCCESS == status && HSA_DEVICE_TYPE_GPU == device_type) {
        hsa_agent_t* ret = (hsa_agent_t*)data;
        *ret = agent;
        return HSA_STATUS_INFO_BREAK;
      }
      return HSA_STATUS_SUCCESS;
    }

    static hsa_status_t region_iterator(hsa_region_t region, void* data) {
      hsa_region_segment_t segment;        
      hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
      
      printf("Segment: %d\t", (int) segment);
      
      hsa_region_global_flag_t flags;
      hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
      
      printf("Flags: %d\t", (int) flags);

      int alloc_allowed;
      hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
      printf("Alloc allowed: %d\n", alloc_allowed);

      return HSA_STATUS_SUCCESS;
    }

    /*static*/ Module *HSAModule::create_module(RuntimeImpl *runtime,
						 std::vector<std::string>& cmdline)
    {
      // before we do anything, make sure there's a CUDA driver and GPUs to talk to
      printf("creating HSA module!\n");

      printf("Initializing HSA Runtime\n");
      hsa_status_t err;

      err = hsa_init();
      if(err != HSA_STATUS_SUCCESS) {
        log_gpu.warning() << "hsa init failed";
        return 0;
      }

      HSAModule *m = new HSAModule;

      printf("Finding HSA agents\n");
      err = hsa_iterate_agents(get_gpu_agent, &m->agent);
      if(err == HSA_STATUS_INFO_BREAK) { err = HSA_STATUS_SUCCESS; }

      if(err != HSA_STATUS_SUCCESS) {
        log_gpu.warning() << "Getting GPU failed";
        return 0;
      }
      
      char name[64] = { 0 };
      err = hsa_agent_get_info(m->agent, HSA_AGENT_INFO_NAME, name);
      printf("The agent name is %s.\n", name);

      hsa_agent_iterate_regions(m->agent, region_iterator, NULL);

      {
	CommandLineParser cp;

	cp.add_option_int("-ll:gpu", m->cfg_num_gpus);
	
	bool ok = cp.parse_command_line(cmdline);
	if(!ok) {
	  log_gpu.error() << "error reading command line options";
	  exit(1);
	}
      }

      return m;      
#if 0
      std::vector<GPUInfo *> infos;
      {
	CUresult ret = cuInit(0);
	if(ret != CUDA_SUCCESS) {
	  log_gpu.warning() << "cuInit(0) returned " << ret << " - module not loaded";
	  return 0;
	}

	int num_devices;
	CHECK_CU( cuDeviceGetCount(&num_devices) );
	for(int i = 0; i < num_devices; i++) {
	  GPUInfo *info = new GPUInfo;

	  // TODO: consider environment variables or other ways to tell if certain
	  //  GPUs should be ignored

	  info->index = i;
	  CHECK_CU( cuDeviceGet(&info->device, i) );
	  CHECK_CU( cuDeviceGetName(info->name, GPUInfo::MAX_NAME_LEN, info->device) );
	  CHECK_CU( cuDeviceComputeCapability(&info->compute_major,
					      &info->compute_minor,
					      info->device) );
	  CHECK_CU( cuDeviceTotalMem(&info->total_mem, info->device) );
	  CHECK_CU( cuDeviceGetProperties(&info->props, info->device) );

	  log_gpu.info() << "GPU #" << i << ": " << info->name << " ("
			 << info->compute_major << '.' << info->compute_minor
			 << ") " << (info->total_mem >> 20) << " MB";

	  infos.push_back(info);
	}

	if(infos.empty()) {
	  log_gpu.warning() << "no CUDA-capable GPUs found - module not loaded";
	  return 0;
	}

	// query peer-to-peer access (all pairs)
	for(std::vector<GPUInfo *>::iterator it1 = infos.begin();
	    it1 != infos.end();
	    it1++)
	  for(std::vector<GPUInfo *>::iterator it2 = infos.begin();
	      it2 != infos.end();
	      it2++)
	    if(it1 != it2) {
	      int can_access;
	      CHECK_CU( cuDeviceCanAccessPeer(&can_access,
					      (*it1)->device,
					      (*it2)->device) );
	      if(can_access)
		(*it1)->peers.insert((*it2)->device);
	    }
      }

      CudaModule *m = new CudaModule;

      // give the gpu info we assembled to the module
      m->gpu_info.swap(infos);

      // first order of business - read command line parameters
      {
	CommandLineParser cp;

	cp.add_option_int("-ll:fsize", m->cfg_fb_mem_size_in_mb)
	  .add_option_int("-ll:zsize", m->cfg_zc_mem_size_in_mb)
	  .add_option_int("-ll:gpu", m->cfg_num_gpus)
	  .add_option_int("-ll:streams", m->cfg_gpu_streams)
	  .add_option_int("-ll:gpuworker", m->cfg_use_shared_worker)
	  .add_option_int("-ll:pin", m->cfg_pin_sysmem)
	  .add_option_bool("-cuda:callbacks", m->cfg_fences_use_callbacks)
	  .add_option_bool("-cuda:nohijack", m->cfg_suppress_hijack_warning);
	
	bool ok = cp.parse_command_line(cmdline);
	if(!ok) {
	  log_gpu.error() << "error reading CUDA command line parameters";
	  exit(1);
	}
      }
#endif
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void HSAModule::initialize(RuntimeImpl *runtime)
    {
      Module::initialize(runtime);

      // sanity-check: do we even have enough gpus?
      if(cfg_num_gpus < 1) {
	log_gpu.fatal() << cfg_num_gpus << " GPUs requested!";
        printf("%d\n", cfg_num_gpus);
	assert(false);
      }


      gpus.resize(cfg_num_gpus);
      for(unsigned i = 0; i < cfg_num_gpus; i++) {
	GPU *g = new GPU(this);

	gpus[i] = g;
      }

#if 0
      // if we are using a shared worker, create that next
      if(cfg_use_shared_worker) {
	shared_worker = new GPUWorker;

	if(cfg_use_background_workers)
	  shared_worker->start_background_thread(runtime->core_reservation_set(),
						 1 << 20); // hardcoded worker stack size
      }

      // just use the GPUs in order right now
      gpus.resize(cfg_num_gpus);
      for(unsigned i = 0; i < cfg_num_gpus; i++) {
	// either create a worker for this GPU or use the shared one
	GPUWorker *worker;
	if(cfg_use_shared_worker) {
	  worker = shared_worker;
	} else {
	  worker = new GPUWorker;

	  if(cfg_use_background_workers)
	    worker->start_background_thread(runtime->core_reservation_set(),
					    1 << 20); // hardcoded worker stack size
	}

	GPU *g = new GPU(this, gpu_info[i], worker, cfg_gpu_streams);

	if(!cfg_use_shared_worker)
	  dedicated_workers[g] = worker;

	gpus[i] = g;
      }
#endif
    }

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    void HSAModule::create_memories(RuntimeImpl *runtime)
    {
      Module::create_memories(runtime);
#if 0
      // each GPU needs its FB memory
      for(std::vector<GPU *>::iterator it = gpus.begin();
	  it != gpus.end();
	  it++)
	(*it)->create_fb_memory(runtime, cfg_fb_mem_size_in_mb << 20);

      // a single ZC memory for everybody
      if((cfg_zc_mem_size_in_mb > 0) && !gpus.empty()) {
	CUdeviceptr zcmem_gpu_base;
	// borrow GPU 0's context for the allocation call
	{
	  AutoGPUContext agc(gpus[0]);

	  CHECK_CU( cuMemHostAlloc(&zcmem_cpu_base, 
				   cfg_zc_mem_size_in_mb << 20,
				   CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP) );
	  CHECK_CU( cuMemHostGetDevicePointer(&zcmem_gpu_base,
					      zcmem_cpu_base,
					      0) );
	  // right now there are asssumptions in several places that unified addressing keeps
	  //  the CPU and GPU addresses the same
	  assert(zcmem_cpu_base == (void *)zcmem_gpu_base);
	}

	Memory m = runtime->next_local_memory_id();
	zcmem = new GPUZCMemory(m, zcmem_gpu_base, zcmem_cpu_base, 
				cfg_zc_mem_size_in_mb << 20);
	runtime->add_memory(zcmem);

	// add the ZC memory as a pinned memory to all GPUs
	for(unsigned i = 0; i < gpus.size(); i++) {
	  CUdeviceptr gpuptr;
	  CUresult ret;
	  {
	    AutoGPUContext agc(gpus[i]);
	    ret = cuMemHostGetDevicePointer(&gpuptr, zcmem_cpu_base, 0);
	  }
	  if((ret == CUDA_SUCCESS) && (gpuptr == zcmem_gpu_base)) {
	    gpus[i]->pinned_sysmems.insert(zcmem->me);
	  } else {
	    log_gpu.warning() << "GPU #" << i << " has an unexpected mapping for ZC memory!";
	  }
	}
      }
#endif
    }

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    void HSAModule::create_processors(RuntimeImpl *runtime)
    {
      Module::create_processors(runtime);

      // each GPU needs a processor
      for(std::vector<GPU *>::iterator it = gpus.begin();
	  it != gpus.end();
	  it++)
	(*it)->create_processor(runtime,
				2 << 20); // TODO: don't use hardcoded stack size...
    }

    // create any DMA channels provided by the module (default == do nothing)
    void HSAModule::create_dma_channels(RuntimeImpl *runtime)
    {
#if 0
      // before we create dma channels, see how many of the system memory ranges
      //  we can register with CUDA
      if(cfg_pin_sysmem && !gpus.empty()) {
	std::vector<MemoryImpl *>& local_mems = runtime->nodes[gasnet_mynode()].memories;
	for(std::vector<MemoryImpl *>::iterator it = local_mems.begin();
	    it != local_mems.end();
	    it++) {
	  // ignore FB/ZC memories or anything that doesn't have a "direct" pointer
	  if(((*it)->kind == MemoryImpl::MKIND_GPUFB) ||
	     ((*it)->kind == MemoryImpl::MKIND_ZEROCOPY))
	    continue;

	  void *base = (*it)->get_direct_ptr(0, (*it)->size);
	  if(base == 0)
	    continue;

	  // using GPU 0's context, attempt a portable registration
	  CUresult ret;
	  {
	    AutoGPUContext agc(gpus[0]);
	    ret = cuMemHostRegister(base, (*it)->size, 
				    CU_MEMHOSTREGISTER_PORTABLE |
				    CU_MEMHOSTREGISTER_DEVICEMAP);
	  }
	  if(ret != CUDA_SUCCESS) {
	    log_gpu.info() << "failed to register mem " << (*it)->me << " (" << base << " + " << (*it)->size << ") : "
			   << ret;
	    continue;
	  }

	  // now go through each GPU and verify that it got a GPU pointer (it may not match the CPU
	  //  pointer, but that's ok because we'll never refer to it directly)
	  for(unsigned i = 0; i < gpus.size(); i++) {
	    CUdeviceptr gpuptr;
	    CUresult ret;
	    {
	      AutoGPUContext agc(gpus[i]);
	      ret = cuMemHostGetDevicePointer(&gpuptr, base, 0);
	    }
	    if(ret == CUDA_SUCCESS) {
	      // no test for && ((void *)gpuptr == base)) {
	      log_gpu.info() << "memory " << (*it)->me << " successfully registered with GPU " << gpus[i]->proc->me;
	      gpus[i]->pinned_sysmems.insert((*it)->me);
	    } else {
	      log_gpu.warning() << "GPU #" << i << " has no mapping for registered memory (" << (*it)->me << " at " << base << ") !?";
	    }
	  }
	}
      }

      // now actually let each GPU make its channels
      for(std::vector<GPU *>::iterator it = gpus.begin();
	  it != gpus.end();
	  it++)
	(*it)->create_dma_channels(runtime);

#endif
      Module::create_dma_channels(runtime);
    }

    // create any code translators provided by the module (default == do nothing)
    void HSAModule::create_code_translators(RuntimeImpl *runtime)
    {
      Module::create_code_translators(runtime);
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void HSAModule::cleanup(void)
    {
      // clean up worker(s)
      if(shared_worker) {
          //	if(cfg_use_background_workers)
          //	  shared_worker->shutdown_background_thread();

	delete shared_worker;
	shared_worker = 0;
      }
      for(std::map<GPU *, GPUWorker *>::iterator it = dedicated_workers.begin();
	  it != dedicated_workers.end();
	  it++) {
	GPUWorker *worker = it->second;

        //	if(cfg_use_background_workers)
        //	  worker->shutdown_background_thread();

	delete worker;
      }
      dedicated_workers.clear();

      for(std::vector<GPU *>::iterator it = gpus.begin();
	  it != gpus.end();
	  it++)
	delete *it;
      gpus.clear();
      
      Module::cleanup();
    }

  }; // namespace HSA
}; // namespace Realm

